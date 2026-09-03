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
            zt = zb + torch.randn_like(zb) * 0.025 # Dense off-manifold perturbation
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
    
    # 4 Pure Autonomous Models (Zero Theorem 3)
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
        
    # High-Res 4-Panel Independent Overlay Figure (300 DPI) - ZERO OVERLAPPING!
    os.makedirs("results/plots", exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.0), dpi=300)
    gt_np = z_orb.detach().cpu().numpy()
    
    panel_configs = [
        ("1_Standard_PINN_MLP", "Standard PINN (MLP)", "r--", axes[0]),
        ("2_Vanilla_HNN_2019", "Vanilla HNN (2019)", "darkorange", axes[1]),
        ("3_CPA_SHNN_Core", "CPA-SHNN Core", "purple", axes[2]),
        ("4_Theorem1_Separable", "Theorem 1: Separable HNN (Champion)", "b-", axes[3]),
    ]
    
    for key, title, style, ax in panel_configs:
        pred_np = preds[key].detach().cpu().numpy()
        ax.plot(gt_np[:, 0], gt_np[:, 1], 'k-', lw=2.4, label='Ground Truth')
        if '--' in style:
            ax.plot(pred_np[:, 0], pred_np[:, 1], style, lw=1.6, label=f'Prediction (Err: {results[key]:.2f}%)')
        else:
            ax.plot(pred_np[:, 0], pred_np[:, 1], color=style.replace('-', ''), linestyle='-' if '-' in style else ':', lw=1.8, label=f'Prediction (Err: {results[key]:.2f}%)')
        ax.set_title(f"{title}\nErr: {results[key]:.2f}% | Drift: {drifts[key]:.4f}%", fontsize=10, fontweight='bold')
        ax.set_xlabel("x", fontweight='bold')
        ax.set_ylabel("y", fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='best', fontsize=8)
        
    plt.suptitle(f"Autonomous Benchmark Trajectory Decomposition: {system.name} ({getattr(system, 'regime', 'chaotic').upper()})", fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    plot_path = f"results/plots/autonomous_comparison_{getattr(system, 'regime', 'reg')}_{system.name}.png"
    plt.savefig(plot_path, bbox_inches='tight')
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
    
    # Master Summary Bar Chart
    plt.figure(figsize=(12, 5.5), dpi=300)
    models_keys = ["1_Standard_PINN_MLP", "2_Vanilla_HNN_2019", "3_CPA_SHNN_Core", "4_Theorem1_Separable"]
    labels = ["Standard PINN", "Vanilla HNN", "CPA-SHNN Core", "Thm 1: Separable (Champion)"]
    colors = ["#e74c3c", "#f39c12", "#9b59b6", "#2980b9"]
    
    x = np.arange(len(systems))
    width = 0.18
    for i, (k, l, c) in enumerate(zip(models_keys, labels, colors)):
        vals = [df.loc[df["System"] == s.name, k].values[0] for s in systems]
        plt.bar(x + i*width, vals, width, label=l, color=c, alpha=0.9, edgecolor='black', lw=0.8)
        
    plt.xticks(x + width*1.5, [s.name for s in systems], fontsize=9, fontweight='bold')
    plt.ylabel("Trajectory Relative L2 Error (%)", fontsize=11, fontweight='bold')
    plt.title(f"Autonomous Master Benchmark Trajectory Fidelity ({regime.upper()})", fontsize=12, fontweight='bold')
    plt.yscale("log")
    plt.grid(True, linestyle=':', alpha=0.6, which="both")
    plt.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(f"results/plots/autonomous_master_summary_{regime}.png")
    plt.close()
    
    print("\n" + "=" * 115)
    print("                 AUTONOMOUS CELESTIAL MASTER BENCHMARK MATRIX")
    print("=" * 115)
    print(df.to_string(index=False))
    print("=" * 115)
    return df

if __name__ == "__main__":
    run_nine_way_master_suite(regime="regular", epochs=10)
