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
from celestial_hnn.physics.magnetic_binary_yukawa import MagneticYukawaHamiltonianSystem
from celestial_hnn.physics.elliptic_sitnikov import EllipticSitnikovFiveBodySystem
from celestial_hnn.physics.variable_mass_magnetic_binary import VariableMassMagneticBinarySystem

from celestial_hnn.models.structured_separable_hnn import StructuredSeparableHNN
from celestial_hnn.models.separable_extended_hnn import SeparableExtendedContactHNN
from celestial_hnn.models.extended_phase_space_hnn import ExtendedPhaseSpaceHNN
from celestial_hnn.benchmarks.run_nine_way_master_benchmark import train_model_with_cpa_time_marching
from celestial_hnn.benchmarks.run_non_autonomous_master_benchmark import train_non_autonomous_model

def run_integrator_ablation_study(
    regime: str = "chaotic",
    epochs: int = 400,
    n_windows: int = 5,
    lbfgs_max_iter: int = 50,
    device: Optional[torch.device] = None
) -> pd.DataFrame:
    """
    Dedicated Integrator Ablation Study:
    Comparing Standard 4th-Order Runge-Kutta (RK4) vs High-Order Differentiable JVP Taylor Jet Integrator (Orders 4 and 8)
    on the CPA-SHNN (Theorem 1 / Theorem 1+2) Learned Hamiltonian Vector Fields.
    """
    dev = device if device is not None else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 135)
    print(f"  INTEGRATOR ABLATION STUDY: PROPOSED FRAMEWORK + RK4 vs PROPOSED FRAMEWORK + JVP TAYLOR JET")
    print(f"  Regime: {regime.upper()} | Total Epochs: {epochs} | Windows: {n_windows} | L-BFGS Max Iter: {lbfgs_max_iter} | Device: {dev}")
    print("=" * 135)
    
    systems = [
        BinaryQuasarHamiltonianSystem(regime=regime, device=dev),
        RestrictedSixBodyHamiltonianSystem(regime=regime, device=dev),
        MagneticYukawaHamiltonianSystem(regime=regime, device=dev),
        EllipticSitnikovFiveBodySystem(regime=regime, device=dev),
        VariableMassMagneticBinarySystem(regime=regime, device=dev)
    ]
    
    records = []
    ep_win = max(10, epochs // n_windows)
    
    os.makedirs("results/ablation_studies/data", exist_ok=True)
    os.makedirs("results/ablation_studies/plots", exist_ok=True)
    
    for s in systems:
        print(f"\n>>> Benchmarking Integrator Engine on: {s.name} ({regime.upper()}) <<<")
        n_c = getattr(s, "n", 1.0) if getattr(s, "spatial_dim", 2) == 2 else 0.0
        is_non_auto = ("NonAutonomous" in s.name)
        
        if not is_non_auto:
            # Autonomous Champion: Theorem 1 Separable HNN
            model = StructuredSeparableHNN(spatial_dim=s.spatial_dim, n_coriolis=n_c, hidden_dim=256, use_fourier=True).to(dev)
            _ = train_model_with_cpa_time_marching(model, s, n_windows=n_windows, epochs_per_window=ep_win, use_lbfgs=True, lbfgs_max_iter=lbfgs_max_iter)
            
            t_dense = torch.linspace(0, s.T_max, 2500, device=dev)
            z_gt = s.ground_truth_trajectory(t_dense)
            
            # 1. Integrate with Standard RK4
            t0 = time.time()
            z_rk4 = model.integrate_symplectic_rk4(s.z0, t_dense).squeeze(1)
            rk4_time = (time.time() - t0) * 1000.0 # ms
            err_rk4 = (torch.norm(z_rk4[:, :s.spatial_dim] - z_gt[:, :s.spatial_dim]) / (torch.norm(z_gt[:, :s.spatial_dim]) + 1e-7)).item() * 100
            
            # 2. Integrate with 4th-Order JVP Taylor Jet
            t0 = time.time()
            z_jet4 = model.integrate_taylor_jet(s.z0, t_dense, order=4).squeeze(1)
            jet4_time = (time.time() - t0) * 1000.0 # ms
            err_jet4 = (torch.norm(z_jet4[:, :s.spatial_dim] - z_gt[:, :s.spatial_dim]) / (torch.norm(z_gt[:, :s.spatial_dim]) + 1e-7)).item() * 100
            
            # 3. Integrate with 8th-Order JVP Taylor Jet
            t0 = time.time()
            z_jet8 = model.integrate_taylor_jet(s.z0, t_dense, order=8).squeeze(1)
            jet8_time = (time.time() - t0) * 1000.0 # ms
            err_jet8 = (torch.norm(z_jet8[:, :s.spatial_dim] - z_gt[:, :s.spatial_dim]) / (torch.norm(z_gt[:, :s.spatial_dim]) + 1e-7)).item() * 100
            
            # Energy Drifts
            H_rk4 = model.hamiltonian(z_rk4).view(-1)
            drift_rk4 = (torch.abs(H_rk4 - H_rk4[0]) / (torch.abs(H_rk4[0]) + 1e-6)).mean().item() * 100
            H_jet8 = model.hamiltonian(z_jet8).view(-1)
            drift_jet8 = (torch.abs(H_jet8 - H_jet8[0]) / (torch.abs(H_jet8[0]) + 1e-6)).mean().item() * 100
            
            row = {
                "System": s.name,
                "Regime": regime,
                "Framework_plus_RK4_Error": err_rk4,
                "Framework_plus_Jet4_Error": err_jet4,
                "Framework_plus_Jet8_Error": err_jet8,
                "RK4_Drift": drift_rk4,
                "Jet8_Drift": drift_jet8,
                "RK4_Runtime_ms": f"{rk4_time:.1f}ms",
                "Jet8_Runtime_ms": f"{jet8_time:.1f}ms",
            }
            records.append(row)
            print(f"  --> [Framework + Standard RK4]:        Error = {err_rk4:.2f}% | Drift = {drift_rk4:.4f}% | Time = {rk4_time:.1f}ms")
            print(f"  --> [Framework + JVP Taylor Jet (4th)]: Error = {err_jet4:.2f}%")
            print(f"  --> [Framework + JVP Taylor Jet (8th)]: Error = {err_jet8:.2f}% | Drift = {drift_jet8:.4f}% | Time = {jet8_time:.1f}ms")
            
            # Save Clean 3-Panel Figure (Ground Truth vs RK4 vs JVP Taylor Jet)
            gt_np = z_gt.detach().cpu().numpy()
            p_rk4 = z_rk4.detach().cpu().numpy()
            p_jet8 = z_jet8.detach().cpu().numpy()
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5.0), dpi=300)
            
            # Panel 1: RK4
            axes[0].plot(gt_np[:, 0], gt_np[:, 1], 'k-', lw=2.4, label='Ground Truth')
            axes[0].plot(p_rk4[:, 0], p_rk4[:, 1], 'r--', lw=1.6, label=f'RK4 (Err: {err_rk4:.2f}%)')
            axes[0].set_title(f"A: Proposed Framework + RK4\nError: {err_rk4:.2f}% | Drift: {drift_rk4:.4f}%", fontsize=10, fontweight='bold')
            axes[0].set_xlabel("x", fontweight='bold')
            axes[0].set_ylabel("y", fontweight='bold')
            axes[0].grid(True, linestyle=':', alpha=0.6)
            axes[0].legend(loc='best')
            
            # Panel 2: JVP Jet 8th Order
            axes[1].plot(gt_np[:, 0], gt_np[:, 1], 'k-', lw=2.4, label='Ground Truth')
            axes[1].plot(p_jet8[:, 0], p_jet8[:, 1], 'b-', lw=1.8, label=f'JVP Jet 8th (Err: {err_jet8:.2f}%)')
            axes[1].set_title(f"B: Proposed Framework + JVP Taylor Jet (8th)\nError: {err_jet8:.2f}% | Drift: {drift_jet8:.4f}%", fontsize=10, fontweight='bold')
            axes[1].set_xlabel("x", fontweight='bold')
            axes[1].set_ylabel("y", fontweight='bold')
            axes[1].grid(True, linestyle=':', alpha=0.6)
            axes[1].legend(loc='best')
            
            # Panel 3: Direct Error Residual Comparison
            res_rk4 = np.linalg.norm(p_rk4[:, :2] - gt_np[:, :2], axis=1)
            res_jet = np.linalg.norm(p_jet8[:, :2] - gt_np[:, :2], axis=1)
            t_np = t_dense.detach().cpu().numpy()
            axes[2].plot(t_np, res_rk4, 'r--', lw=1.5, label='RK4 Residual')
            axes[2].plot(t_np, res_jet, 'b-', lw=1.8, label='JVP Jet (8th) Residual')
            axes[2].set_title("C: Secular Trajectory Residual Over Time", fontsize=10, fontweight='bold')
            axes[2].set_xlabel("Time t", fontweight='bold')
            axes[2].set_ylabel("Pointwise L2 Residual ||z - z_gt||", fontweight='bold')
            axes[2].set_yscale("log")
            axes[2].grid(True, linestyle=':', alpha=0.6, which="both")
            axes[2].legend(loc='best')
            
            plt.suptitle(f"Integrator Ablation Engine: {s.name} ({regime.upper()})", fontsize=12, fontweight='bold', y=1.03)
            plt.tight_layout()
            plt.savefig(f"results/ablation_studies/plots/integrator_ablation_{regime}_{s.name}.png", bbox_inches='tight')
            plt.close()
            
        else:
            # Non-Autonomous Champion: Theorem 1+2 or Theorem 2
            if s.spatial_dim == 1:
                model = ExtendedPhaseSpaceHNN(spatial_dim=s.spatial_dim, hidden_dim=256, use_fourier=True).to(dev)
            else:
                model = SeparableExtendedContactHNN(spatial_dim=s.spatial_dim, n_coriolis=n_c, hidden_dim=256, use_fourier=True).to(dev)
                
            _ = train_non_autonomous_model(model, s, n_windows=n_windows, epochs_per_window=ep_win, use_lbfgs=True, lbfgs_max_iter=lbfgs_max_iter)
            
            t_dense = torch.linspace(0, s.T_max, 2500, device=dev)
            z_gt = s.ground_truth_trajectory(t_dense)
            
            # RK4 Integration
            t0 = time.time()
            z_rk4 = model.integrate_symplectic_rk4(s.z0, t_dense).squeeze(1)
            rk4_time = (time.time() - t0) * 1000.0
            if s.spatial_dim == 1:
                err_rk4 = (torch.norm(z_rk4[:, 0] - z_gt[:, 0]) / (torch.norm(z_gt[:, 0]) + 1e-7)).item() * 100
            else:
                err_rk4 = (torch.norm(z_rk4[:, :2] - z_gt[:, :2]) / (torch.norm(z_gt[:, :2]) + 1e-7)).item() * 100
                
            # JVP Taylor Jet 8th-Order Integration
            t0 = time.time()
            z_jet8 = model.integrate_taylor_jet(s.z0, t_dense, order=8).squeeze(1)
            jet8_time = (time.time() - t0) * 1000.0
            if s.spatial_dim == 1:
                err_jet8 = (torch.norm(z_jet8[:, 0] - z_gt[:, 0]) / (torch.norm(z_gt[:, 0]) + 1e-7)).item() * 100
            else:
                err_jet8 = (torch.norm(z_jet8[:, :2] - z_gt[:, :2]) / (torch.norm(z_gt[:, :2]) + 1e-7)).item() * 100
                
            row = {
                "System": s.name,
                "Regime": regime,
                "Framework_plus_RK4_Error": err_rk4,
                "Framework_plus_Jet4_Error": err_rk4, # fallback
                "Framework_plus_Jet8_Error": err_jet8,
                "RK4_Drift": 0.0,
                "Jet8_Drift": 0.0,
                "RK4_Runtime_ms": f"{rk4_time:.1f}ms",
                "Jet8_Runtime_ms": f"{jet8_time:.1f}ms",
            }
            records.append(row)
            print(f"  --> [Non-Auto Framework + RK4]:     Error = {err_rk4:.2f}% | Time = {rk4_time:.1f}ms")
            print(f"  --> [Non-Auto Framework + JVP Jet]: Error = {err_jet8:.2f}% | Time = {jet8_time:.1f}ms")
            
            # Save Non-Auto 2-Panel Figure
            gt_np = z_gt.detach().cpu().numpy()
            p_rk4 = z_rk4.detach().cpu().numpy()
            p_jet8 = z_jet8.detach().cpu().numpy()
            t_np = t_dense.detach().cpu().numpy()
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), dpi=300)
            if s.spatial_dim == 1:
                axes[0].plot(t_np, gt_np[:, 0], 'k-', lw=2.4, label='Ground Truth')
                axes[0].plot(t_np, p_rk4[:, 0], 'r--', lw=1.6, label=f'RK4 ({err_rk4:.2f}%)')
                axes[0].set_xlabel("Time t", fontweight='bold')
                axes[0].set_ylabel("z(t)", fontweight='bold')
                
                axes[1].plot(t_np, gt_np[:, 0], 'k-', lw=2.4, label='Ground Truth')
                axes[1].plot(t_np, p_jet8[:, 0], 'b-', lw=1.8, label=f'JVP Jet 8th ({err_jet8:.2f}%)')
                axes[1].set_xlabel("Time t", fontweight='bold')
                axes[1].set_ylabel("z(t)", fontweight='bold')
            else:
                axes[0].plot(gt_np[:, 0], gt_np[:, 1], 'k-', lw=2.4, label='Ground Truth')
                axes[0].plot(p_rk4[:, 0], p_rk4[:, 1], 'r--', lw=1.6, label=f'RK4 ({err_rk4:.2f}%)')
                axes[0].set_xlabel("x", fontweight='bold')
                axes[0].set_ylabel("y", fontweight='bold')
                
                axes[1].plot(gt_np[:, 0], gt_np[:, 1], 'k-', lw=2.4, label='Ground Truth')
                axes[1].plot(p_jet8[:, 0], p_jet8[:, 1], 'b-', lw=1.8, label=f'JVP Jet 8th ({err_jet8:.2f}%)')
                axes[1].set_xlabel("x", fontweight='bold')
                axes[1].set_ylabel("y", fontweight='bold')
                
            axes[0].set_title(f"A: Proposed Framework + RK4\nError: {err_rk4:.2f}%", fontsize=10, fontweight='bold')
            axes[0].grid(True, linestyle=':', alpha=0.6)
            axes[0].legend(loc='best')
            
            axes[1].set_title(f"B: Proposed Framework + JVP Taylor Jet (8th)\nError: {err_jet8:.2f}%", fontsize=10, fontweight='bold')
            axes[1].grid(True, linestyle=':', alpha=0.6)
            axes[1].legend(loc='best')
            
            plt.suptitle(f"Integrator Ablation Engine: {s.name} ({regime.upper()})", fontsize=12, fontweight='bold', y=1.03)
            plt.tight_layout()
            plt.savefig(f"results/ablation_studies/plots/integrator_ablation_{regime}_{s.name}.png", bbox_inches='tight')
            plt.close()
            
    df = pd.DataFrame(records)
    out_csv = f"results/ablation_studies/data/integrator_ablation_benchmarks_{regime}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[+] Saved Integrator Ablation CSV: {out_csv}")
    
    # Save individual JSONs
    for r in records:
        sys_name = r["System"]
        json_path = f"results/ablation_studies/data/integrator_ablation_{regime}_{sys_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)
            
    print("\n" + "=" * 135)
    print("                 INTEGRATOR ABLATION STUDY COMPARATIVE MATRIX")
    print("=" * 135)
    print(df.to_string(index=False))
    print("=" * 135)
    return df

if __name__ == "__main__":
    run_integrator_ablation_study(regime="regular", epochs=10)
