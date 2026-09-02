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
from celestial_hnn.models.extended_phase_space_hnn import ExtendedPhaseSpaceHNN
from celestial_hnn.models.generating_function_hnn import NeuralSymplecticGeneratingMap
from celestial_hnn.models.separable_generating_hnn import SeparableGeneratingMapHNN
from celestial_hnn.models.extended_generating_hnn import ExtendedGeneratingMapHNN
from celestial_hnn.models.grand_unified_engine import GrandUnifiedSymplecticEngine
from celestial_hnn.training.adaptive_time_marching_trainer import AdaptiveTimeMarchingHNNTrainer

def train_and_evaluate_model(model, system, z_orb, dz_true, H_exact, H0, t_dense, epochs=400, lr=3e-3, is_time_dependent=False):
    dev = z_orb.device
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    
    for _ in range(epochs):
        opt.zero_grad()
        idx = torch.randint(0, len(z_orb), (min(1024, len(z_orb)),), device=dev)
        zb = z_orb[idx]
        zt = zb + torch.randn_like(zb) * 0.03
        za = torch.cat([zb, zt], dim=0)
        dza = torch.cat([dz_true[idx], system.canonical_derivatives(zt)], dim=0)
        
        dz_p = model.time_derivative(za)
        loss_f = torch.mean((dz_p - dza) ** 2)
        H_p = model.hamiltonian(zb)
        loss_h = torch.mean((H_p - H_exact[idx]) ** 2)
        
        loss = loss_f + 1.5 * loss_h
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        
    err = system.compute_trajectory_error(model)
    z_pred = model.integrate_symplectic_rk4(system.z0, t_dense).squeeze(1)
    H_pred = model.hamiltonian(z_pred)
    drift = (torch.abs(H_pred - H0) / (torch.abs(H0) + 1e-6)).mean().item() * 100
    return err * 100, drift, z_pred

def run_nine_way_single_system(system, epochs: int = 400, device: Optional[torch.device] = None) -> Dict[str, Any]:
    dev = device if device is not None else system.device
    t_dense = torch.linspace(0, system.T_max, 2500, device=dev)
    z_orb = system.ground_truth_trajectory(t_dense)
    dz_true = system.canonical_derivatives(z_orb)
    H_exact = system.exact_hamiltonian(z_orb)
    H0 = system.exact_hamiltonian(z_orb[0:1])
    n_c = getattr(system, "n", 1.0) if system.spatial_dim == 2 else 0.0
    
    results = {}
    preds = {}
    
    # 1. Standard Vector Field MLP
    mlp = BaselineVectorFieldMLP(state_dim=2*system.spatial_dim, hidden_dim=256).to(dev)
    opt_mlp = torch.optim.AdamW(mlp.parameters(), lr=3e-3)
    for _ in range(epochs):
        opt_mlp.zero_grad()
        idx = torch.randint(0, len(z_orb), (min(1024, len(z_orb)),), device=dev)
        loss = torch.mean((mlp(z_orb[idx]) - dz_true[idx]) ** 2)
        loss.backward()
        opt_mlp.step()
    err_mlp = system.compute_trajectory_error(mlp)
    z_mlp = mlp.integrate_symplectic_rk4(system.z0, t_dense).squeeze(1)
    results["1_Standard_PINN_MLP"] = err_mlp * 100
    preds["1_Standard_PINN_MLP"] = z_mlp
    
    # 2. Vanilla HNN (2019)
    vhnn = HamiltonianNeuralNetwork(spatial_dim=system.spatial_dim, hidden_dim=256).to(dev)
    err_vhnn, drift_vhnn, z_vhnn = train_and_evaluate_model(vhnn, system, z_orb, dz_true, H_exact, H0, t_dense, epochs=epochs)
    results["2_Vanilla_HNN_2019"] = err_vhnn
    preds["2_Vanilla_HNN_2019"] = z_vhnn
    
    # 3. CPA-SHNN (Proposed Core Time-Marching)
    cpa_trainer = AdaptiveTimeMarchingHNNTrainer(system=system, n_windows=4, hidden_dim=256, device=dev)
    cpa_model, cpa_res = cpa_trainer.train_adaptive_time_marching(epochs_per_window=max(10, epochs//4), use_lbfgs=True, lbfgs_max_iter=30, verbose=False)
    results["3_CPA_SHNN_Core"] = cpa_res["rel_l2_error"] * 100
    preds["3_CPA_SHNN_Core"] = cpa_model.integrate_symplectic_rk4(system.z0, t_dense).squeeze(1)
    
    # 4. Theorem 1: Separable-HNN
    m_thm1 = StructuredSeparableHNN(spatial_dim=system.spatial_dim, n_coriolis=n_c, hidden_dim=256).to(dev)
    err_t1, drift_t1, z_t1 = train_and_evaluate_model(m_thm1, system, z_orb, dz_true, H_exact, H0, t_dense, epochs=epochs)
    results["4_Theorem1_Separable"] = err_t1
    preds["4_Theorem1_Separable"] = z_t1
    
    # 5. Theorem 2: Extended Contact Space
    m_thm2 = ExtendedPhaseSpaceHNN(spatial_dim=system.spatial_dim, hidden_dim=256).to(dev)
    err_t2, drift_t2, z_t2 = train_and_evaluate_model(m_thm2, system, z_orb, dz_true, H_exact, H0, t_dense, epochs=epochs, is_time_dependent=True)
    results["5_Theorem2_ExtendedSpace"] = err_t2
    preds["5_Theorem2_ExtendedSpace"] = z_t2
    
    # 6. Theorem 3: Symplectic Generating Map
    m_thm3 = NeuralSymplecticGeneratingMap(spatial_dim=system.spatial_dim, hidden_dim=256).to(dev)
    err_t3, drift_t3, z_t3 = train_and_evaluate_model(m_thm3, system, z_orb, dz_true, H_exact, H0, t_dense, epochs=epochs)
    results["6_Theorem3_GeneratingMap"] = err_t3
    preds["6_Theorem3_GeneratingMap"] = z_t3
    
    # 7. Combo 1+3: Separable Generating Map
    m_c13 = SeparableGeneratingMapHNN(spatial_dim=system.spatial_dim, n_coriolis=n_c, hidden_dim=256).to(dev)
    err_c13, drift_c13, z_c13 = train_and_evaluate_model(m_c13, system, z_orb, dz_true, H_exact, H0, t_dense, epochs=epochs)
    results["7_Combo_1_plus_3"] = err_c13
    preds["7_Combo_1_plus_3"] = z_c13
    
    # 8. Combo 2+3: Extended Generating Map
    m_c23 = ExtendedGeneratingMapHNN(spatial_dim=system.spatial_dim, hidden_dim=256).to(dev)
    err_c23, drift_c23, z_c23 = train_and_evaluate_model(m_c23, system, z_orb, dz_true, H_exact, H0, t_dense, epochs=epochs, is_time_dependent=True)
    results["8_Combo_2_plus_3"] = err_c23
    preds["8_Combo_2_plus_3"] = z_c23
    
    # 9. Combo 1+2+3: Grand Unified Symplectic Engine
    m_c123 = GrandUnifiedSymplecticEngine(spatial_dim=system.spatial_dim, n_coriolis=n_c, hidden_dim=256).to(dev)
    err_c123, drift_c123, z_c123 = train_and_evaluate_model(m_c123, system, z_orb, dz_true, H_exact, H0, t_dense, epochs=epochs, is_time_dependent=True)
    results["9_Combo_1_2_3_Unified"] = err_c123
    preds["9_Combo_1_2_3_Unified"] = z_c123
    
    # Plot Visual Comparison
    os.makedirs("results/plots", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    
    # Panel 1: Ground Truth vs MLP vs Vanilla
    gt_np = z_orb.detach().cpu().numpy()
    mlp_np = z_mlp.detach().cpu().numpy()
    vhnn_np = z_vhnn.detach().cpu().numpy()
    axes[0].plot(gt_np[:, 0], gt_np[:, 1], 'k-', lw=2.5, label='Ground Truth')
    axes[0].plot(mlp_np[:, 0], mlp_np[:, 1], 'r--', lw=1.5, alpha=0.8, label=f'Standard MLP ({err_mlp*100:.1f}%)')
    axes[0].plot(vhnn_np[:, 0], vhnn_np[:, 1], 'g:', lw=1.5, alpha=0.8, label=f'Vanilla HNN ({err_vhnn:.1f}%)')
    axes[0].set_title(f"A: Baseline Baselines\n{system.name}", fontsize=11, fontweight='bold')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='best', fontsize=8)
    
    # Panel 2: The Core Theorems (1, 2, 3)
    t1_np = z_t1.detach().cpu().numpy()
    t2_np = z_t2.detach().cpu().numpy()
    t3_np = z_t3.detach().cpu().numpy()
    axes[1].plot(gt_np[:, 0], gt_np[:, 1], 'k-', lw=2.5, label='Ground Truth')
    axes[1].plot(t1_np[:, 0], t1_np[:, 1], 'b-', lw=1.8, label=f'Thm 1: Separable ({err_t1:.2f}%)')
    axes[1].plot(t2_np[:, 0], t2_np[:, 1], 'm--', lw=1.2, alpha=0.7, label=f'Thm 2: Extended ({err_t2:.1f}%)')
    axes[1].plot(t3_np[:, 0], t3_np[:, 1], 'c:', lw=1.2, alpha=0.7, label=f'Thm 3: Generating ({err_t3:.1f}%)')
    axes[1].set_title(f"B: Theoretical Foundations\n(Theorems 1, 2, 3)", fontsize=11, fontweight='bold')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='best', fontsize=8)
    
    # Panel 3: The Unified Combos (1+3, 2+3, 1+2+3) vs CPA-SHNN
    cpa_np = preds["3_CPA_SHNN_Core"].detach().cpu().numpy()
    u_np = z_c123.detach().cpu().numpy()
    axes[2].plot(gt_np[:, 0], gt_np[:, 1], 'k-', lw=2.5, label='Ground Truth')
    axes[2].plot(cpa_np[:, 0], cpa_np[:, 1], 'green', lw=2.0, label=f'CPA-SHNN Core ({cpa_res["rel_l2_error"]*100:.2f}%)')
    axes[2].plot(u_np[:, 0], u_np[:, 1], 'red', linestyle='--', lw=1.5, alpha=0.8, label=f'Grand Unified 1+2+3 ({err_c123:.2f}%)')
    axes[2].set_title(f"C: Grand Unified Synthesis\n(CPA-SHNN & Unified Engine)", fontsize=11, fontweight='bold')
    axes[2].grid(True, linestyle=':', alpha=0.6)
    axes[2].legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    plot_path = f"results/plots/nine_way_comparison_{getattr(system, 'regime', 'reg')}_{system.name}.png"
    plt.savefig(plot_path)
    plt.close()
    
    # Save System JSON
    os.makedirs("results/data", exist_ok=True)
    json_path = f"results/data/nine_way_{getattr(system, 'regime', 'reg')}_{system.name}_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    return results

def run_nine_way_master_suite(regime: str = "chaotic", epochs: int = 400, device: Optional[torch.device] = None) -> pd.DataFrame:
    dev = device if device is not None else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 95)
    print(f"  NINE-WAY GRAND SCIENTIFIC BENCHMARK MASTER SUITE")
    print(f"  Regime: {regime.upper()} | Epochs: {epochs} | Device: {dev}")
    print("=" * 95)
    
    systems = [
        BinaryQuasarHamiltonianSystem(regime=regime, device=dev),
        RestrictedSixBodyHamiltonianSystem(regime=regime, device=dev),
        SitnikovFiveBodyHamiltonianSystem(regime=regime, device=dev),
        MagneticYukawaHamiltonianSystem(regime=regime, device=dev),
    ]
    
    records = []
    for s in systems:
        print(f"\n--- 9-Way Benchmark for System: {s.name} ({regime}) ---")
        t0 = time.time()
        res = run_nine_way_single_system(s, epochs=epochs, device=dev)
        el = time.time() - t0
        res["System"] = s.name
        res["Regime"] = regime
        res["Runtime_s"] = f"{el:.1f}s"
        records.append(res)
        print(f"  [+] Complete in {el:.1f}s | Thm1 (Sep): {res['4_Theorem1_Separable']:.2f}% | CPA-Core: {res['3_CPA_SHNN_Core']:.2f}% | Unified: {res['9_Combo_1_2_3_Unified']:.2f}%")
        
    df = pd.DataFrame(records)
    cols = ["System", "Regime", "1_Standard_PINN_MLP", "2_Vanilla_HNN_2019", "3_CPA_SHNN_Core", "4_Theorem1_Separable", "5_Theorem2_ExtendedSpace", "6_Theorem3_GeneratingMap", "7_Combo_1_plus_3", "8_Combo_2_plus_3", "9_Combo_1_2_3_Unified", "Runtime_s"]
    df = df[cols]
    
    out_csv = f"results/data/nine_way_master_benchmarks_{regime}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[+] Saved 9-Way Master CSV: {out_csv}")
    
    # Master Bar Chart
    plt.figure(figsize=(14, 6), dpi=300)
    models_keys = ["1_Standard_PINN_MLP", "2_Vanilla_HNN_2019", "3_CPA_SHNN_Core", "4_Theorem1_Separable", "9_Combo_1_2_3_Unified"]
    labels = ["Standard PINN", "Vanilla HNN", "CPA-SHNN (Core)", "Thm 1 (Separable)", "Grand Unified Engine"]
    
    x = np.arange(len(systems))
    width = 0.15
    for i, (k, l) in enumerate(zip(models_keys, labels)):
        vals = [df.loc[df["System"] == s.name, k].values[0] for s in systems]
        plt.bar(x + i*width, vals, width, label=l)
        
    plt.xticks(x + width*2, [s.name for s in systems], fontsize=9)
    plt.ylabel("Trajectory Relative L2 Error (%)", fontsize=11, fontweight='bold')
    plt.title(f"9-Way Architectural Paradigm Benchmark ({regime.upper()})", fontsize=13, fontweight='bold')
    plt.yscale("log")
    plt.grid(True, linestyle=':', alpha=0.6, which="both")
    plt.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(f"results/plots/nine_way_master_summary_{regime}.png")
    plt.close()
    
    print("\n" + "=" * 115)
    print("                      NINE-WAY GRAND SCIENTIFIC BENCHMARK MATRIX")
    print("=" * 115)
    print(df.to_string(index=False))
    print("=" * 115)
    
    return df

if __name__ == "__main__":
    run_nine_way_master_suite(regime="regular", epochs=5)
