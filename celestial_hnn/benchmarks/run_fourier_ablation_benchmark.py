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
from celestial_hnn.models.extended_phase_space_hnn import ExtendedPhaseSpaceHNN
from celestial_hnn.models.separable_extended_hnn import SeparableExtendedContactHNN
from celestial_hnn.benchmarks.run_nine_way_master_benchmark import train_model_with_cpa_time_marching
from celestial_hnn.benchmarks.run_non_autonomous_master_benchmark import train_non_autonomous_model

def run_fourier_ablation_study(
    regime: str = "chaotic",
    epochs: int = 400,
    n_windows: int = 5,
    lbfgs_max_iter: int = 50,
    device: Optional[torch.device] = None
) -> pd.DataFrame:
    """
    Conducts a rigorous ablation study comparing Pure Smooth C^infinity MLPs vs Multi-Scale Fourier Positional Encodings
    across representative Autonomous and Non-Autonomous Celestial Systems.
    """
    dev = device if device is not None else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 125)
    print(f"  FOURIER POSITIONAL ENCODING ABLATION SUITE (WITH VS WITHOUT FOURIER)")
    print(f"  Regime: {regime.upper()} | Total Epochs: {epochs} | Windows: {n_windows} | L-BFGS Max Iter: {lbfgs_max_iter} | Device: {dev}")
    print("=" * 125)
    
    systems = [
        BinaryQuasarHamiltonianSystem(regime=regime, device=dev),
        RestrictedSixBodyHamiltonianSystem(regime=regime, device=dev),
        MagneticYukawaHamiltonianSystem(regime=regime, device=dev),
        EllipticSitnikovFiveBodySystem(regime=regime, device=dev),
        VariableMassMagneticBinarySystem(regime=regime, device=dev)
    ]
    
    records = []
    ep_win = max(10, epochs // n_windows)
    
    for s in systems:
        print(f"\n>>> Benchmarking Fourier Ablation on: {s.name} ({regime.upper()}) <<<")
        n_c = getattr(s, "n", 1.0) if getattr(s, "spatial_dim", 2) == 2 else 0.0
        is_non_auto = ("NonAutonomous" in s.name)
        
        if not is_non_auto:
            # Autonomous: Theorem 1 (No Fourier vs With Fourier)
            m_no_fourier = StructuredSeparableHNN(spatial_dim=s.spatial_dim, n_coriolis=n_c, hidden_dim=256, use_fourier=False).to(dev)
            m_with_fourier = StructuredSeparableHNN(spatial_dim=s.spatial_dim, n_coriolis=n_c, hidden_dim=256, use_fourier=True).to(dev)
            
            res_no = train_model_with_cpa_time_marching(m_no_fourier, s, n_windows=n_windows, epochs_per_window=ep_win, use_lbfgs=True, lbfgs_max_iter=lbfgs_max_iter)
            res_with = train_model_with_cpa_time_marching(m_with_fourier, s, n_windows=n_windows, epochs_per_window=ep_win, use_lbfgs=True, lbfgs_max_iter=lbfgs_max_iter)
            
            row = {
                "System": s.name,
                "Regime": regime,
                "No_Fourier_Error": res_no["rel_l2_error"],
                "With_Fourier_Error": res_with["rel_l2_error"],
                "Improvement_Factor": f"{res_no['rel_l2_error'] / max(1e-3, res_with['rel_l2_error']):.2f}x"
            }
            records.append(row)
            print(f"  --> [No Fourier]:   Error = {res_no['rel_l2_error']:.2f}% | Drift = {res_no['energy_drift']:.4f}%")
            print(f"  --> [With Fourier]: Error = {res_with['rel_l2_error']:.2f}% | Drift = {res_with['energy_drift']:.4f}% (Gain: {row['Improvement_Factor']})")
            
            # Clean 2-Panel Plot (No Fourier vs With Fourier)
            os.makedirs("results/plots", exist_ok=True)
            t_dense = torch.linspace(0, s.T_max, 3000, device=dev)
            gt_np = s.ground_truth_trajectory(t_dense).detach().cpu().numpy()
            p_no = res_no["z_pred"].detach().cpu().numpy()
            p_with = res_with["z_pred"].detach().cpu().numpy()
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)
            
            axes[0].plot(gt_np[:, 0], gt_np[:, 1], 'k-', lw=2.4, label='Ground Truth')
            axes[0].plot(p_no[:, 0], p_no[:, 1], 'r--', lw=1.6, label=f'No Fourier (Err: {res_no["rel_l2_error"]:.2f}%)')
            axes[0].set_title(f"A: Without Fourier Features\nError: {res_no['rel_l2_error']:.2f}%", fontsize=10, fontweight='bold')
            axes[0].set_xlabel("x", fontweight='bold')
            axes[0].set_ylabel("y", fontweight='bold')
            axes[0].grid(True, linestyle=':', alpha=0.6)
            axes[0].legend(loc='best')
            
            axes[1].plot(gt_np[:, 0], gt_np[:, 1], 'k-', lw=2.4, label='Ground Truth')
            axes[1].plot(p_with[:, 0], p_with[:, 1], 'b-', lw=1.8, label=f'With Fourier (Err: {res_with["rel_l2_error"]:.2f}%)')
            axes[1].set_title(f"B: With Multi-Scale Fourier\nError: {res_with['rel_l2_error']:.2f}% ({row['Improvement_Factor']} Gain)", fontsize=10, fontweight='bold')
            axes[1].set_xlabel("x", fontweight='bold')
            axes[1].set_ylabel("y", fontweight='bold')
            axes[1].grid(True, linestyle=':', alpha=0.6)
            axes[1].legend(loc='best')
            
            plt.suptitle(f"Fourier Encoding Ablation: {s.name} ({regime.upper()})", fontsize=12, fontweight='bold', y=1.03)
            plt.tight_layout()
            plt.savefig(f"results/plots/fourier_ablation_{regime}_{s.name}.png", bbox_inches='tight')
            plt.close()
            
        else:
            # Non-Autonomous: Elliptic Sitnikov (Thm 2) or Variable Mass (Thm 1+2)
            if s.spatial_dim == 1:
                m_no = ExtendedPhaseSpaceHNN(spatial_dim=s.spatial_dim, hidden_dim=256, use_fourier=False).to(dev)
                m_with = ExtendedPhaseSpaceHNN(spatial_dim=s.spatial_dim, hidden_dim=256, use_fourier=True).to(dev)
            else:
                m_no = SeparableExtendedContactHNN(spatial_dim=s.spatial_dim, n_coriolis=n_c, hidden_dim=256, use_fourier=False).to(dev)
                m_with = SeparableExtendedContactHNN(spatial_dim=s.spatial_dim, n_coriolis=n_c, hidden_dim=256, use_fourier=True).to(dev)
                
            res_no = train_non_autonomous_model(m_no, s, n_windows=n_windows, epochs_per_window=ep_win, use_lbfgs=True, lbfgs_max_iter=lbfgs_max_iter)
            res_with = train_non_autonomous_model(m_with, s, n_windows=n_windows, epochs_per_window=ep_win, use_lbfgs=True, lbfgs_max_iter=lbfgs_max_iter)
            
            row = {
                "System": s.name,
                "Regime": regime,
                "No_Fourier_Error": res_no["rel_l2_error"],
                "With_Fourier_Error": res_with["rel_l2_error"],
                "Improvement_Factor": f"{res_no['rel_l2_error'] / max(1e-3, res_with['rel_l2_error']):.2f}x"
            }
            records.append(row)
            print(f"  --> [No Fourier]:   Error = {res_no['rel_l2_error']:.2f}%")
            print(f"  --> [With Fourier]: Error = {res_with['rel_l2_error']:.2f}% (Gain: {row['Improvement_Factor']})")
            
            # Clean 2-Panel Plot
            os.makedirs("results/plots", exist_ok=True)
            t_dense = torch.linspace(0, s.T_max, 2500, device=dev)
            gt_np = s.ground_truth_trajectory(t_dense).detach().cpu().numpy()
            p_no = res_no["z_pred"].detach().cpu().numpy()
            p_with = res_with["z_pred"].detach().cpu().numpy()
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)
            if s.spatial_dim == 1:
                t_np = t_dense.detach().cpu().numpy()
                axes[0].plot(t_np, gt_np[:, 0], 'k-', lw=2.4, label='Ground Truth')
                axes[0].plot(t_np, p_no[:, 0], 'r--', lw=1.6, label=f'No Fourier ({res_no["rel_l2_error"]:.2f}%)')
                axes[0].set_xlabel("Time t", fontweight='bold')
                axes[0].set_ylabel("z(t)", fontweight='bold')
                
                axes[1].plot(t_np, gt_np[:, 0], 'k-', lw=2.4, label='Ground Truth')
                axes[1].plot(t_np, p_with[:, 0], 'b-', lw=1.8, label=f'With Fourier ({res_with["rel_l2_error"]:.2f}%)')
                axes[1].set_xlabel("Time t", fontweight='bold')
                axes[1].set_ylabel("z(t)", fontweight='bold')
            else:
                axes[0].plot(gt_np[:, 0], gt_np[:, 1], 'k-', lw=2.4, label='Ground Truth')
                axes[0].plot(p_no[:, 0], p_no[:, 1], 'r--', lw=1.6, label=f'No Fourier ({res_no["rel_l2_error"]:.2f}%)')
                axes[0].set_xlabel("x", fontweight='bold')
                axes[0].set_ylabel("y", fontweight='bold')
                
                axes[1].plot(gt_np[:, 0], gt_np[:, 1], 'k-', lw=2.4, label='Ground Truth')
                axes[1].plot(p_with[:, 0], p_with[:, 1], 'b-', lw=1.8, label=f'With Fourier ({res_with["rel_l2_error"]:.2f}%)')
                axes[1].set_xlabel("x", fontweight='bold')
                axes[1].set_ylabel("y", fontweight='bold')
                
            axes[0].set_title(f"A: Without Fourier Features\nError: {res_no['rel_l2_error']:.2f}%", fontsize=10, fontweight='bold')
            axes[0].grid(True, linestyle=':', alpha=0.6)
            axes[0].legend(loc='best')
            
            axes[1].set_title(f"B: With Multi-Scale Fourier\nError: {res_with['rel_l2_error']:.2f}% ({row['Improvement_Factor']} Gain)", fontsize=10, fontweight='bold')
            axes[1].grid(True, linestyle=':', alpha=0.6)
            axes[1].legend(loc='best')
            
            plt.suptitle(f"Fourier Encoding Ablation: {s.name} ({regime.upper()})", fontsize=12, fontweight='bold', y=1.03)
            plt.tight_layout()
            plt.savefig(f"results/plots/fourier_ablation_{regime}_{s.name}.png", bbox_inches='tight')
            plt.close()

    df = pd.DataFrame(records)
    out_csv = f"results/data/fourier_ablation_benchmarks_{regime}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[+] Saved Fourier Ablation CSV: {out_csv}")
    
    print("\n" + "=" * 115)
    print("                 FOURIER POSITIONAL ENCODING ABLATION MATRIX")
    print("=" * 115)
    print(df.to_string(index=False))
    print("=" * 115)
    return df

if __name__ == "__main__":
    run_fourier_ablation_study(regime="regular", epochs=10)
