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
from celestial_hnn.models.generating_function_hnn import NeuralSymplecticGeneratingMap
from celestial_hnn.models.separable_generating_hnn import SeparableGeneratingMapHNN
from celestial_hnn.models.separable_extended_hnn import SeparableExtendedContactHNN
from celestial_hnn.models.extended_generating_hnn import ExtendedGeneratingMapHNN
from celestial_hnn.models.grand_unified_engine import GrandUnifiedSymplecticEngine

from celestial_hnn.benchmarks.run_nine_way_master_benchmark import train_model_with_cpa_time_marching
from celestial_hnn.benchmarks.run_non_autonomous_master_benchmark import train_non_autonomous_model

def run_generating_function_ablation_study(
    regime: str = "chaotic",
    epochs: int = 400,
    n_windows: int = 5,
    lbfgs_max_iter: int = 50,
    device: Optional[torch.device] = None
) -> pd.DataFrame:
    """
    Dedicated Empirical Ablation Study:
    Investigating Poincaré-Jacobi Generating Function Neural Networks (Theorem 3 & Combos)
    vs Continuous Vector Field Formulations (Theorems 1 & 2).
    """
    dev = device if device is not None else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 135)
    print(f"  POINCARÉ GENERATING FUNCTION ABLATION STUDY (CONTINUOUS HNN VS DISCRETE GENERATING MAPS)")
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
    
    # Dedicated ablation directories
    os.makedirs("results/ablation_studies/data", exist_ok=True)
    os.makedirs("results/ablation_studies/plots", exist_ok=True)
    
    for s in systems:
        print(f"\n>>> Benchmarking Generating Map Ablation on: {s.name} ({regime.upper()}) <<<")
        n_c = getattr(s, "n", 1.0) if getattr(s, "spatial_dim", 2) == 2 else 0.0
        is_non_auto = ("NonAutonomous" in s.name)
        
        if not is_non_auto:
            # Autonomous Comparison: Continuous Theorem 1 vs Pure Generating Map vs Combo 1+3
            models = {
                "Continuous_Theorem1_Separable": StructuredSeparableHNN(spatial_dim=s.spatial_dim, n_coriolis=n_c, hidden_dim=256, use_fourier=True).to(dev),
                "Discrete_Theorem3_GeneratingMap": NeuralSymplecticGeneratingMap(spatial_dim=s.spatial_dim, hidden_dim=256, use_fourier=True).to(dev),
                "Discrete_Combo13_SeparableGen": SeparableGeneratingMapHNN(spatial_dim=s.spatial_dim, n_coriolis=n_c, hidden_dim=256, use_fourier=True).to(dev),
            }
            res_dict = {}
            preds_dict = {}
            for name, m in models.items():
                r = train_model_with_cpa_time_marching(m, s, n_windows=n_windows, epochs_per_window=ep_win, use_lbfgs=True, lbfgs_max_iter=lbfgs_max_iter)
                res_dict[name] = r["rel_l2_error"]
                preds_dict[name] = r["z_pred"]
                print(f"  --> [{name}]: Error = {r['rel_l2_error']:.2f}% | Drift = {r['energy_drift']:.4f}%")
                
            row = {
                "System": s.name,
                "Regime": regime,
                "Continuous_Thm1_Err": res_dict["Continuous_Theorem1_Separable"],
                "Discrete_Thm3_Err": res_dict["Discrete_Theorem3_GeneratingMap"],
                "Combo13_SepGen_Err": res_dict["Discrete_Combo13_SeparableGen"],
            }
            records.append(row)
            
            # Save 3-Panel Independent Comparison Figure
            t_dense = torch.linspace(0, s.T_max, 3000, device=dev)
            gt_np = s.ground_truth_trajectory(t_dense).detach().cpu().numpy()
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5.0), dpi=300)
            panels = [
                ("Continuous_Theorem1_Separable", "Continuous Theorem 1 (Champion)", "b-", axes[0]),
                ("Discrete_Theorem3_GeneratingMap", "Discrete Theorem 3 (Poincaré Map)", "r--", axes[1]),
                ("Discrete_Combo13_SeparableGen", "Combo 1+3 (Separable Generating Map)", "purple", axes[2]),
            ]
            for key, title, style, ax in panels:
                p_np = preds_dict[key].detach().cpu().numpy()
                ax.plot(gt_np[:, 0], gt_np[:, 1], 'k-', lw=2.4, label='Ground Truth')
                if '--' in style:
                    ax.plot(p_np[:, 0], p_np[:, 1], style, lw=1.5, label=f'Pred (Err: {res_dict[key]:.2f}%)')
                else:
                    ax.plot(p_np[:, 0], p_np[:, 1], color=style.replace('-', ''), lw=1.8, label=f'Pred (Err: {res_dict[key]:.2f}%)')
                ax.set_title(f"{title}\nError: {res_dict[key]:.2f}%", fontsize=10, fontweight='bold')
                ax.set_xlabel("x", fontweight='bold')
                ax.set_ylabel("y", fontweight='bold')
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend(loc='best', fontsize=8)
                
            plt.suptitle(f"Ablation Analysis: Continuous Vector Fields vs Discrete Generating Maps ({s.name})", fontsize=12, fontweight='bold', y=1.03)
            plt.tight_layout()
            plt.savefig(f"results/ablation_studies/plots/generating_map_ablation_{regime}_{s.name}.png", bbox_inches='tight')
            plt.close()
            
        else:
            # Non-Autonomous Comparison: Continuous Theorem 1+2 vs Discrete Combo 2+3 vs Combo 1+2+3
            models = {
                "Continuous_Theorem1_plus_2": SeparableExtendedContactHNN(spatial_dim=s.spatial_dim, n_coriolis=n_c, hidden_dim=256, use_fourier=True).to(dev),
                "Discrete_Combo23_ExtendedGen": ExtendedGeneratingMapHNN(spatial_dim=s.spatial_dim, hidden_dim=256, use_fourier=True).to(dev),
                "Discrete_Combo123_GrandUnified": GrandUnifiedSymplecticEngine(spatial_dim=s.spatial_dim, n_coriolis=n_c, hidden_dim=256, use_fourier=True).to(dev),
            }
            res_dict = {}
            preds_dict = {}
            for name, m in models.items():
                r = train_non_autonomous_model(m, s, n_windows=n_windows, epochs_per_window=ep_win, use_lbfgs=True, lbfgs_max_iter=lbfgs_max_iter)
                res_dict[name] = r["rel_l2_error"]
                preds_dict[name] = r["z_pred"]
                print(f"  --> [{name}]: Error = {r['rel_l2_error']:.2f}%")
                
            row = {
                "System": s.name,
                "Regime": regime,
                "Continuous_Thm1_plus_2_Err": res_dict["Continuous_Theorem1_plus_2"],
                "Discrete_Combo23_Err": res_dict["Discrete_Combo23_ExtendedGen"],
                "Discrete_Combo123_Err": res_dict["Discrete_Combo123_GrandUnified"],
            }
            records.append(row)
            
            # Save 3-Panel Independent Comparison Figure
            t_dense = torch.linspace(0, s.T_max, 2500, device=dev)
            gt_np = s.ground_truth_trajectory(t_dense).detach().cpu().numpy()
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5.0), dpi=300)
            panels = [
                ("Continuous_Theorem1_plus_2", "Continuous Theorem 1+2 (Champion)", "b-", axes[0]),
                ("Discrete_Combo23_ExtendedGen", "Discrete Combo 2+3 (Extended Gen)", "r--", axes[1]),
                ("Discrete_Combo123_GrandUnified", "Grand Unified (1+2+3)", "purple", axes[2]),
            ]
            for key, title, style, ax in panels:
                p_np = preds_dict[key].detach().cpu().numpy()
                if s.spatial_dim == 1:
                    t_np = t_dense.detach().cpu().numpy()
                    ax.plot(t_np, gt_np[:, 0], 'k-', lw=2.4, label='Ground Truth')
                    if '--' in style:
                        ax.plot(t_np, p_np[:, 0], style, lw=1.5, label=f'Pred (Err: {res_dict[key]:.2f}%)')
                    else:
                        ax.plot(t_np, p_np[:, 0], color=style.replace('-', ''), lw=1.8, label=f'Pred (Err: {res_dict[key]:.2f}%)')
                    ax.set_xlabel("Time t", fontweight='bold')
                    ax.set_ylabel("z(t)", fontweight='bold')
                else:
                    ax.plot(gt_np[:, 0], gt_np[:, 1], 'k-', lw=2.4, label='Ground Truth')
                    if '--' in style:
                        ax.plot(p_np[:, 0], p_np[:, 1], style, lw=1.5, label=f'Pred (Err: {res_dict[key]:.2f}%)')
                    else:
                        ax.plot(p_np[:, 0], p_np[:, 1], color=style.replace('-', ''), lw=1.8, label=f'Pred (Err: {res_dict[key]:.2f}%)')
                    ax.set_xlabel("x", fontweight='bold')
                    ax.set_ylabel("y", fontweight='bold')
                    
                ax.set_title(f"{title}\nError: {res_dict[key]:.2f}%", fontsize=10, fontweight='bold')
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend(loc='best', fontsize=8)
                
            plt.suptitle(f"Ablation Analysis: Continuous Vector Fields vs Discrete Generating Maps ({s.name})", fontsize=12, fontweight='bold', y=1.03)
            plt.tight_layout()
            plt.savefig(f"results/ablation_studies/plots/generating_map_ablation_{regime}_{s.name}.png", bbox_inches='tight')
            plt.close()
            
    df = pd.DataFrame(records)
    out_csv = f"results/ablation_studies/data/generating_map_ablation_benchmarks_{regime}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[+] Saved Generating Map Ablation CSV: {out_csv}")
    
    # Save individual JSONs
    for r in records:
        sys_name = r["System"]
        json_path = f"results/ablation_studies/data/generating_map_ablation_{regime}_{sys_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)
            
    print("\n" + "=" * 135)
    print("                 GENERATING FUNCTION ABLATION STUDY COMPARATIVE MATRIX")
    print("=" * 135)
    print(df.to_string(index=False))
    print("=" * 135)
    return df

if __name__ == "__main__":
    run_generating_function_ablation_study(regime="regular", epochs=10)
