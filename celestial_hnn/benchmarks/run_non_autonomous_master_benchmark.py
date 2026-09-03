import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import json
import os
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional

from celestial_hnn.physics.elliptic_sitnikov import EllipticSitnikovFiveBodySystem
from celestial_hnn.physics.variable_mass_magnetic_binary import VariableMassMagneticBinarySystem

from celestial_hnn.models.baseline_mlp import BaselineVectorFieldMLP
from celestial_hnn.models.hnn import HamiltonianNeuralNetwork
from celestial_hnn.models.structured_separable_hnn import StructuredSeparableHNN
from celestial_hnn.models.extended_phase_space_hnn import ExtendedPhaseSpaceHNN
from celestial_hnn.models.separable_extended_hnn import SeparableExtendedContactHNN

def train_non_autonomous_model(
    model,
    system,
    n_windows: int = 4,
    epochs_per_window: int = 60,
    use_lbfgs: bool = True,
    lbfgs_max_iter: int = 40
) -> Dict[str, Any]:
    dev = system.device
    T_max = system.T_max
    window_boundaries = torch.linspace(0, T_max, n_windows + 1, device=dev)
    
    for w in range(n_windows):
        t_curr_end = window_boundaries[w + 1]
        t_w = torch.linspace(0, t_curr_end, 800 * (w + 1), device=dev)
        z_w = system.ground_truth_trajectory(t_w)
        dz_w = system.canonical_derivatives(z_w)
        
        opt_adam = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-6)
        
        for _ in range(epochs_per_window):
            opt_adam.zero_grad()
            idx = torch.randint(0, len(z_w), (min(512, len(z_w)),), device=dev)
            zb = z_w[idx]
            zt = zb + torch.randn_like(zb) * 0.02
            za = torch.cat([zb, zt], dim=0)
            dza = torch.cat([dz_w[idx], system.canonical_derivatives(zt)], dim=0)
            
            dz_p = model.time_derivative(za) if hasattr(model, "time_derivative") else model(za)
            loss = torch.mean((dz_p - dza) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt_adam.step()
            
        if use_lbfgs:
            opt_lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.8, max_iter=lbfgs_max_iter, history_size=25, line_search_fn="strong_wolfe")
            zb_l = z_w
            dza_l = dz_w
            def closure():
                opt_lbfgs.zero_grad()
                dz_p = model.time_derivative(zb_l) if hasattr(model, "time_derivative") else model(zb_l)
                loss = torch.mean((dz_p - dza_l) ** 2)
                loss.backward()
                return loss
            try:
                opt_lbfgs.step(closure)
            except Exception:
                pass
                
    err = system.compute_trajectory_error(model)
    t_dense = torch.linspace(0, T_max, 2500, device=dev)
    z_pred = model.integrate_symplectic_rk4(system.z0, t_dense).squeeze(1)
    return {"rel_l2_error": err * 100, "z_pred": z_pred}

def run_non_autonomous_master_suite(
    regime: str = "chaotic",
    epochs: int = 240,
    n_windows: int = 4,
    device: Optional[torch.device] = None
) -> pd.DataFrame:
    dev = device if device is not None else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 110)
    print(f"  NON-AUTONOMOUS CELESTIAL BENCHMARK SUITE (PROF. VINAY KUMAR SYSTEMS)")
    print(f"  Regime: {regime.upper()} | Epochs: {epochs} | Windows: {n_windows} | Device: {dev}")
    print("=" * 110)
    
    systems = [
        EllipticSitnikovFiveBodySystem(regime=regime, device=dev),
        VariableMassMagneticBinarySystem(regime=regime, device=dev)
    ]
    
    records = []
    ep_win = max(10, epochs // n_windows)
    
    for s in systems:
        print(f"\n>>> Benchmarking Non-Autonomous System: {s.name} ({regime.upper()}) <<<")
        t0 = time.time()
        n_c = getattr(s, "n", 1.0) if s.spatial_dim == 2 else 0.0
        
        models = {
            "1_Standard_PINN_MLP": BaselineVectorFieldMLP(state_dim=s.state_dim, hidden_dim=256).to(dev),
            "2_Theorem2_ExtendedContactHNN": ExtendedPhaseSpaceHNN(spatial_dim=s.spatial_dim, hidden_dim=256).to(dev),
            "3_Theorem1_plus_2_SeparableExtendedHNN": SeparableExtendedContactHNN(spatial_dim=s.spatial_dim, n_coriolis=n_c, hidden_dim=256).to(dev)
        }
        
        res_dict = {}
        preds_dict = {}
        for name, m in models.items():
            res = train_non_autonomous_model(m, s, n_windows=n_windows, epochs_per_window=ep_win, use_lbfgs=True, lbfgs_max_iter=30)
            res_dict[name] = res["rel_l2_error"]
            preds_dict[name] = res["z_pred"]
            print(f"  --> Model [{name}]: Error = {res['rel_l2_error']:.2f}%")
            
        el = time.time() - t0
        res_dict["System"] = s.name
        res_dict["Regime"] = regime
        res_dict["Runtime_s"] = f"{el:.1f}s"
        records.append(res_dict)
        
        # High-Res 2-Panel Trajectory Plot (300 DPI)
        os.makedirs("results/plots", exist_ok=True)
        t_dense = torch.linspace(0, s.T_max, 2500, device=dev)
        z_gt = s.ground_truth_trajectory(t_dense).detach().cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
        
        if s.spatial_dim == 1:
            # Sitnikov vertical oscillation z(t)
            t_np = t_dense.detach().cpu().numpy()
            axes[0].plot(t_np, z_gt[:, 0], 'k-', lw=2.5, label='Ground Truth')
            axes[0].plot(t_np, preds_dict["1_Standard_PINN_MLP"].detach().cpu().numpy()[:, 0], 'r--', lw=1.5, label=f'Standard MLP ({res_dict["1_Standard_PINN_MLP"]:.1f}%)')
            axes[0].plot(t_np, preds_dict["3_Theorem1_plus_2_SeparableExtendedHNN"].detach().cpu().numpy()[:, 0], 'b-', lw=1.8, label=f'Separable Extended HNN ({res_dict["3_Theorem1_plus_2_SeparableExtendedHNN"]:.2f}%)')
            axes[0].set_xlabel("Time t", fontweight='bold')
            axes[0].set_ylabel("Vertical Position z(t)", fontweight='bold')
            axes[0].set_title(f"A: Non-Autonomous Sitnikov Oscillation\n({s.name})", fontsize=11, fontweight='bold')
            axes[0].grid(True, linestyle=':', alpha=0.6)
            axes[0].legend(loc='best')
            
            # Phase portrait (z, pz)
            axes[1].plot(z_gt[:, 0], z_gt[:, 2], 'k-', lw=2.5, label='Ground Truth')
            axes[1].plot(preds_dict["1_Standard_PINN_MLP"].detach().cpu().numpy()[:, 0], preds_dict["1_Standard_PINN_MLP"].detach().cpu().numpy()[:, 2], 'r--', lw=1.5, label='MLP')
            axes[1].plot(preds_dict["3_Theorem1_plus_2_SeparableExtendedHNN"].detach().cpu().numpy()[:, 0], preds_dict["3_Theorem1_plus_2_SeparableExtendedHNN"].detach().cpu().numpy()[:, 2], 'b-', lw=1.8, label='Separable Extended HNN')
            axes[1].set_xlabel("z", fontweight='bold')
            axes[1].set_ylabel("pz", fontweight='bold')
            axes[1].set_title("B: Non-Autonomous Phase Portrait (z, pz)", fontsize=11, fontweight='bold')
            axes[1].grid(True, linestyle=':', alpha=0.6)
            axes[1].legend(loc='best')
        else:
            # Variable Mass (x, y) plane
            axes[0].plot(z_gt[:, 0], z_gt[:, 1], 'k-', lw=2.5, label='Ground Truth')
            axes[0].plot(preds_dict["1_Standard_PINN_MLP"].detach().cpu().numpy()[:, 0], preds_dict["1_Standard_PINN_MLP"].detach().cpu().numpy()[:, 1], 'r--', lw=1.5, label=f'Standard MLP ({res_dict["1_Standard_PINN_MLP"]:.1f}%)')
            axes[0].plot(preds_dict["3_Theorem1_plus_2_SeparableExtendedHNN"].detach().cpu().numpy()[:, 0], preds_dict["3_Theorem1_plus_2_SeparableExtendedHNN"].detach().cpu().numpy()[:, 1], 'b-', lw=1.8, label=f'Separable Extended HNN ({res_dict["3_Theorem1_plus_2_SeparableExtendedHNN"]:.2f}%)')
            axes[0].set_xlabel("x", fontweight='bold')
            axes[0].set_ylabel("y", fontweight='bold')
            axes[0].set_title(f"A: Variable-Mass Trajectory (x, y)\n({s.name})", fontsize=11, fontweight='bold')
            axes[0].grid(True, linestyle=':', alpha=0.6)
            axes[0].legend(loc='best')
            
            # (x, px) Phase Portrait
            axes[1].plot(z_gt[:, 0], z_gt[:, 3], 'k-', lw=2.5, label='Ground Truth')
            axes[1].plot(preds_dict["3_Theorem1_plus_2_SeparableExtendedHNN"].detach().cpu().numpy()[:, 0], preds_dict["3_Theorem1_plus_2_SeparableExtendedHNN"].detach().cpu().numpy()[:, 3], 'b-', lw=1.8, label='Separable Extended HNN')
            axes[1].set_xlabel("x", fontweight='bold')
            axes[1].set_ylabel("px", fontweight='bold')
            axes[1].set_title("B: Contact Phase Portrait (x, px)", fontsize=11, fontweight='bold')
            axes[1].grid(True, linestyle=':', alpha=0.6)
            axes[1].legend(loc='best')
            
        plt.tight_layout()
        plot_path = f"results/plots/non_autonomous_{getattr(s, 'regime', 'reg')}_{s.name}.png"
        plt.savefig(plot_path)
        plt.close()
        
    df = pd.DataFrame(records)
    cols = ["System", "Regime", "1_Standard_PINN_MLP", "2_Theorem2_ExtendedContactHNN", "3_Theorem1_plus_2_SeparableExtendedHNN", "Runtime_s"]
    df = df[cols]
    
    os.makedirs("results/data", exist_ok=True)
    out_csv = f"results/data/non_autonomous_master_benchmarks_{regime}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[+] Saved Non-Autonomous Master CSV: {out_csv}")
    
    print("\n" + "=" * 110)
    print("                 NON-AUTONOMOUS CELESTIAL BENCHMARK MATRIX")
    print("=" * 110)
    print(df.to_string(index=False))
    print("=" * 110)
    
    return df

if __name__ == "__main__":
    run_non_autonomous_master_suite(regime="regular", epochs=10)
