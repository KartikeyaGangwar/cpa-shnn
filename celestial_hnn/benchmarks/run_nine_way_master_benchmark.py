import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import json
import os
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional

from celestial_hnn.physics.binary_quasar import BinaryQuasarHamiltonianSystem
from celestial_hnn.physics.restricted_six_body import RestrictedSixBodyHamiltonianSystem
from celestial_hnn.physics.sitnikov_five_body import SitnikovFiveBodyHamiltonianSystem
from celestial_hnn.physics.magnetic_binary_yukawa import MagneticYukawaHamiltonianSystem

from celestial_hnn.models.baseline_mlp import BaselineVectorFieldMLP
from celestial_hnn.models.hnn import HamiltonianNeuralNetwork
from celestial_hnn.models.structured_separable_hnn import StructuredSeparableHNN

def train_model_with_cpa_time_marching(
    model: nn.Module, 
    system: Any, 
    n_windows: int = 5, 
    epochs_per_window: int = 80, 
    use_lbfgs: bool = True, 
    lbfgs_max_iter: int = 50,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Autonomous CPA Time-Marching Engine:
    Curriculum causal windows + AdamW (Cosine Annealing) + Aggressive L-BFGS (Strong-Wolfe).
    """
    dev = system.device
    T_max = system.T_max
    window_boundaries = torch.linspace(0, T_max, n_windows + 1, device=dev)
    
    for w in range(n_windows):
        t_curr_end = window_boundaries[w + 1]
        t_w = torch.linspace(0, t_curr_end, 1000 * (w + 1), device=dev)
        z_w = system.ground_truth_trajectory(t_w)
        dz_w = system.canonical_derivatives(z_w)
        H_w = system.exact_hamiltonian(z_w)
        
        opt_adam = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_adam, T_max=epochs_per_window, eta_min=1e-5)
        
        for ep in range(epochs_per_window):
            opt_adam.zero_grad()
            idx = torch.randint(0, len(z_w), (min(1024, len(z_w)),), device=dev)
            zb = z_w[idx]
            zt = zb + torch.randn_like(zb) * 0.025 # Dense off-manifold collocation
            za = torch.cat([zb, zt], dim=0)
            dza = torch.cat([dz_w[idx], system.canonical_derivatives(zt)], dim=0)
            
            dz_p = model.time_derivative(za) if hasattr(model, "time_derivative") else model(za)
            loss_f = torch.mean((dz_p - dza) ** 2)
            if hasattr(model, "hamiltonian"):
                H_p = model.hamiltonian(zb)
                loss_h = torch.mean((H_p - H_w[idx]) ** 2)
                loss = loss_f + 1.5 * loss_h
            else:
                loss = loss_f
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt_adam.step()
            scheduler.step()
            
        if use_lbfgs:
            opt_lbfgs = torch.optim.LBFGS(
                model.parameters(), 
                lr=0.8, 
                max_iter=lbfgs_max_iter, 
                history_size=30, 
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-7,
                tolerance_change=1e-9
            )
            zb_l = z_w
            dza_l = dz_w
            H_w_l = H_w
            def closure():
                opt_lbfgs.zero_grad()
                dz_p = model.time_derivative(zb_l) if hasattr(model, "time_derivative") else model(zb_l)
                loss_f = torch.mean((dz_p - dza_l) ** 2)
                if hasattr(model, "hamiltonian"):
                    H_p = model.hamiltonian(zb_l)
                    loss_h = torch.mean((H_p - H_w_l) ** 2)
                    loss = loss_f + 1.5 * loss_h
                else:
                    loss = loss_f
                loss.backward()
                return loss
            try:
                opt_lbfgs.step(closure)
            except Exception:
                pass
                
    err = system.compute_trajectory_error(model)
    t_dense = torch.linspace(0, T_max, 3000, device=dev)
    z_pred = model.integrate_symplectic_rk4(system.z0, t_dense).squeeze(1)
    if hasattr(model, "hamiltonian"):
        H_pred = model.hamiltonian(z_pred).view(-1)
        H0 = H_pred[0]
        drift = (torch.abs(H_pred - H0) / (torch.abs(H0) + 1e-6)).mean().item() * 100
    else:
        drift = 100.0
    return {"rel_l2_error": err * 100, "energy_drift": drift, "z_pred": z_pred}

def run_nine_way_single_system(
    system, 
    epochs: int = 400, 
    n_windows: int = 5, 
    lbfgs_max_iter: int = 50,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    dev = device if device is not None else system.device
    t_dense = torch.linspace(0, system.T_max, 3000, device=dev)
    z_orb = system.ground_truth_trajectory(t_dense)
    n_c = getattr(system, "n", 1.0) if system.spatial_dim == 2 else 0.0
    ep_win = max(10, epochs // n_windows)
    
    # 4 Pure Autonomous Models (Theorem 1 as Champion)
    models = {
        "1_Standard_PINN_MLP": BaselineVectorFieldMLP(state_dim=2*system.spatial_dim, hidden_dim=256).to(dev),
        "2_Vanilla_HNN_2019": HamiltonianNeuralNetwork(spatial_dim=system.spatial_dim, hidden_dim=256, use_fourier=True).to(dev),
        "3_CPA_SHNN_Core": StructuredSeparableHNN(spatial_dim=system.spatial_dim, n_coriolis=n_c, hidden_dim=256, use_fourier=True).to(dev),
        "4_Theorem1_Separable": StructuredSeparableHNN(spatial_dim=system.spatial_dim, n_coriolis=n_c, hidden_dim=256, use_fourier=True).to(dev),
    }
    
    results = {}
    drifts = {}
    preds = {}
    
    for name, m in models.items():
        res = train_model_with_cpa_time_marching(
            m, system, n_windows=n_windows, epochs_per_window=ep_win, use_lbfgs=True, lbfgs_max_iter=lbfgs_max_iter
        )
        results[name] = res["rel_l2_error"]
        drifts[name] = res["energy_drift"]
        preds[name] = res["z_pred"]
        print(f"  --> Model [{name}]: Error = {res['rel_l2_error']:.2f}% | Drift = {res['energy_drift']:.4f}%")
        
    # High-Res 2-Panel Trajectory Plot (300 DPI)
    os.makedirs("results/plots", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    gt_np = z_orb.detach().cpu().numpy()
    
    # Panel 1: Trajectory Comparison
    axes[0].plot(gt_np[:, 0], gt_np[:, 1], 'k-', lw=2.5, label='Ground Truth')
    axes[0].plot(preds["1_Standard_PINN_MLP"].detach().cpu().numpy()[:, 0], preds["1_Standard_PINN_MLP"].detach().cpu().numpy()[:, 1], 'r--', lw=1.2, label=f'Standard PINN ({results["1_Standard_PINN_MLP"]:.1f}%)')
    axes[0].plot(preds["2_Vanilla_HNN_2019"].detach().cpu().numpy()[:, 0], preds["2_Vanilla_HNN_2019"].detach().cpu().numpy()[:, 1], 'gray', linestyle=':', lw=1.2, label=f'Vanilla HNN ({results["2_Vanilla_HNN_2019"]:.1f}%)')
    axes[0].plot(preds["4_Theorem1_Separable"].detach().cpu().numpy()[:, 0], preds["4_Theorem1_Separable"].detach().cpu().numpy()[:, 1], 'b-', lw=2.0, label=f'Theorem 1 Separable ({results["4_Theorem1_Separable"]:.2f}%)')
    axes[0].set_title(f"A: Trajectory Configuration\n{system.name}", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("x", fontweight='bold')
    axes[0].set_ylabel("y", fontweight='bold')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='best', fontsize=8)
    
    # Panel 2: Phase Space (x, px)
    axes[1].plot(gt_np[:, 0], gt_np[:, system.spatial_dim], 'k-', lw=2.5, label='Ground Truth')
    axes[1].plot(preds["4_Theorem1_Separable"].detach().cpu().numpy()[:, 0], preds["4_Theorem1_Separable"].detach().cpu().numpy()[:, system.spatial_dim], 'b-', lw=2.0, label=f'Theorem 1 (Drift={drifts["4_Theorem1_Separable"]:.4f}%)')
    axes[1].set_title(f"B: Phase Space (x, px)\n{system.name}", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("x", fontweight='bold')
    axes[1].set_ylabel("px", fontweight='bold')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    plot_path = f"results/plots/autonomous_comparison_{getattr(system, 'regime', 'reg')}_{system.name}.png"
    plt.savefig(plot_path)
    plt.close()
    
    # Save Full System JSON
    os.makedirs("results/data", exist_ok=True)
    json_path = f"results/data/autonomous_{getattr(system, 'regime', 'reg')}_{system.name}_results.json"
    full_export = {"errors": results, "energy_drifts": drifts}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_export, f, indent=2)
        
    return results

def run_nine_way_master_suite(
    regime: str = "chaotic", 
    epochs: int = 400, 
    n_windows: int = 5, 
    lbfgs_max_iter: int = 50,
    device: Optional[torch.device] = None
) -> pd.DataFrame:
    dev = device if device is not None else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 115)
    print(f"  AUTONOMOUS CELESTIAL MASTER BENCHMARK SUITE (4 CANONICAL SYSTEMS)")
    print(f"  Regime: {regime.upper()} | Total Epochs: {epochs} | Windows: {n_windows} | L-BFGS Max Iter: {lbfgs_max_iter} | Device: {dev}")
    print("=" * 115)
    
    systems = [
        BinaryQuasarHamiltonianSystem(regime=regime, device=dev),
        RestrictedSixBodyHamiltonianSystem(regime=regime, device=dev),
        SitnikovFiveBodyHamiltonianSystem(regime=regime, device=dev),
        MagneticYukawaHamiltonianSystem(regime=regime, device=dev),
    ]
    
    records = []
    for s in systems:
        print(f"\n>>> Benchmarking Autonomous System: {s.name} ({regime.upper()}) <<<")
        t0 = time.time()
        res = run_nine_way_single_system(s, epochs=epochs, n_windows=n_windows, lbfgs_max_iter=lbfgs_max_iter, device=dev)
        el = time.time() - t0
        res["System"] = s.name
        res["Regime"] = regime
        res["Runtime_s"] = f"{el:.1f}s"
        records.append(res)
        
    df = pd.DataFrame(records)
    cols = ["System", "Regime", "1_Standard_PINN_MLP", "2_Vanilla_HNN_2019", "3_CPA_SHNN_Core", "4_Theorem1_Separable", "Runtime_s"]
    df = df[cols]
    
    out_csv = f"results/data/autonomous_master_benchmarks_{regime}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[+] Saved Autonomous Master CSV: {out_csv}")
    
    print("\n" + "=" * 115)
    print("                 AUTONOMOUS CELESTIAL MASTER BENCHMARK MATRIX")
    print("=" * 115)
    print(df.to_string(index=False))
    print("=" * 115)
    return df

if __name__ == "__main__":
    run_nine_way_master_suite(regime="regular", epochs=10)
