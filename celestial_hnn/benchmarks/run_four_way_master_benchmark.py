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

def run_four_way_system_benchmark(
    system,
    epochs: int = 400,
    use_lbfgs: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Runs 4-Way Grand Scientific Benchmark on a single celestial system:
      1. Baseline Vector Field MLP
      2. Vanilla HNN (Greydanus 2019)
      3. Structured Separable HNN (Theorem 1)
      4. Extended / Generating Symplectic Engine (Theorem 2 / Theorem 3)
    """
    dev = device if device is not None else system.device
    
    t_dense = torch.linspace(0, system.T_max, 2500, device=dev)
    z_orb = system.ground_truth_trajectory(t_dense)
    dz_true = system.canonical_derivatives(z_orb)
    H_exact = system.exact_hamiltonian(z_orb)
    H0 = system.exact_hamiltonian(z_orb[0:1])
    
    results = {}
    
    # -------------------------------------------------------------
    # 1. Baseline Vector Field MLP
    # -------------------------------------------------------------
    mlp = BaselineVectorFieldMLP(state_dim=2*system.spatial_dim, hidden_dim=256).to(dev)
    opt_mlp = torch.optim.AdamW(mlp.parameters(), lr=3e-3)
    for _ in range(epochs):
        opt_mlp.zero_grad()
        idx = torch.randint(0, len(z_orb), (min(1024, len(z_orb)),), device=dev)
        loss = torch.mean((mlp(z_orb[idx]) - dz_true[idx]) ** 2)
        loss.backward()
        opt_mlp.step()
    err_mlp = system.compute_trajectory_error(mlp)
    results["Standard_MLP_Rel_L2"] = err_mlp * 100
    
    # -------------------------------------------------------------
    # 2. Vanilla HNN (Greydanus 2019)
    # -------------------------------------------------------------
    vanilla_hnn = HamiltonianNeuralNetwork(spatial_dim=system.spatial_dim, hidden_dim=256).to(dev)
    opt_vhnn = torch.optim.AdamW(vanilla_hnn.parameters(), lr=3e-3)
    for _ in range(epochs):
        opt_vhnn.zero_grad()
        idx = torch.randint(0, len(z_orb), (min(1024, len(z_orb)),), device=dev)
        dz_pred = vanilla_hnn.time_derivative(z_orb[idx])
        loss = torch.mean((dz_pred - dz_true[idx]) ** 2)
        loss.backward()
        opt_vhnn.step()
    err_vhnn = system.compute_trajectory_error(vanilla_hnn)
    z_pred_vhnn = vanilla_hnn.integrate_symplectic_rk4(system.z0, t_dense).squeeze(1)
    H_pred_vhnn = vanilla_hnn.hamiltonian(z_pred_vhnn)
    drift_vhnn = (torch.abs(H_pred_vhnn - H0) / (torch.abs(H0) + 1e-6)).mean().item() * 100
    results["Vanilla_HNN_Rel_L2"] = err_vhnn * 100
    results["Vanilla_HNN_Drift"] = drift_vhnn
    
    # -------------------------------------------------------------
    # 3. Structured Separable HNN (Theorem 1)
    # -------------------------------------------------------------
    n_c = getattr(system, "n", 1.0) if system.spatial_dim == 2 else 0.0
    sep_hnn = StructuredSeparableHNN(spatial_dim=system.spatial_dim, n_coriolis=n_c, hidden_dim=256).to(dev)
    opt_sep = torch.optim.AdamW(sep_hnn.parameters(), lr=3e-3)
    for _ in range(epochs):
        opt_sep.zero_grad()
        idx = torch.randint(0, len(z_orb), (min(1024, len(z_orb)),), device=dev)
        zb = z_orb[idx]
        zt = zb + torch.randn_like(zb) * 0.03
        za = torch.cat([zb, zt], dim=0)
        dza = torch.cat([dz_true[idx], system.canonical_derivatives(zt)], dim=0)
        dz_p = sep_hnn.time_derivative(za)
        loss_f = torch.mean((dz_p - dza) ** 2)
        H_p = sep_hnn.hamiltonian(zb)
        loss_h = torch.mean((H_p - H_exact[idx]) ** 2)
        tot = loss_f + 2.0 * loss_h
        tot.backward()
        opt_sep.step()
    if use_lbfgs:
        lbfgs = torch.optim.LBFGS(sep_hnn.parameters(), lr=0.5, max_iter=40, history_size=30, line_search_fn="strong_wolfe")
        def cl():
            lbfgs.zero_grad()
            dz_p = sep_hnn.time_derivative(z_orb)
            lf = torch.mean((dz_p - dz_true)**2)
            hp = sep_hnn.hamiltonian(z_orb)
            lh = torch.mean((hp - H_exact)**2)
            tot = lf + 2.0 * lh
            tot.backward()
            return tot
        lbfgs.step(cl)
    err_sep = system.compute_trajectory_error(sep_hnn)
    z_pred_sep = sep_hnn.integrate_symplectic_rk4(system.z0, t_dense).squeeze(1)
    H_pred_sep = sep_hnn.hamiltonian(z_pred_sep)
    drift_sep = (torch.abs(H_pred_sep - H0) / (torch.abs(H0) + 1e-6)).mean().item() * 100
    results["Theorem1_Separable_Rel_L2"] = err_sep * 100
    results["Theorem1_Separable_Drift"] = drift_sep
    
    # -------------------------------------------------------------
    # 4. Extended Space HNN / Generating Map (Theorem 2 / 3)
    # -------------------------------------------------------------
    if system.spatial_dim == 1:
        # Sitnikov non-autonomous case -> Use Theorem 2
        ext_hnn = ExtendedPhaseSpaceHNN(spatial_dim=1, hidden_dim=256).to(dev)
        opt_ext = torch.optim.AdamW(ext_hnn.parameters(), lr=3e-3)
        t_col = t_dense.unsqueeze(-1)
        for _ in range(epochs):
            opt_ext.zero_grad()
            idx = torch.randint(0, len(z_orb), (min(1024, len(z_orb)),), device=dev)
            zb = z_orb[idx]
            dz_p = ext_hnn.time_derivative(zb)
            loss_f = torch.mean((dz_p - dz_true[idx]) ** 2)
            H_p = ext_hnn.hamiltonian(zb)
            loss_h = torch.mean((H_p - H_exact[idx]) ** 2)
            tot = loss_f + 2.0 * loss_h
            tot.backward()
            opt_ext.step()
        err_adv = system.compute_trajectory_error(ext_hnn)
        z_pred_adv = ext_hnn.integrate_symplectic_rk4(system.z0, t_dense).squeeze(1)
        H_pred_adv = ext_hnn.hamiltonian(z_pred_adv)
        drift_adv = (torch.abs(H_pred_adv - H0) / (torch.abs(H0) + 1e-6)).mean().item() * 100
    else:
        # 2D systems -> Use Theorem 3 Generating Function
        gen_map = NeuralSymplecticGeneratingMap(spatial_dim=system.spatial_dim, hidden_dim=256).to(dev)
        opt_gen = torch.optim.AdamW(gen_map.parameters(), lr=3e-3)
        for _ in range(epochs):
            opt_gen.zero_grad()
            idx = torch.randint(0, len(z_orb), (min(1024, len(z_orb)),), device=dev)
            zb = z_orb[idx]
            dz_p = gen_map.time_derivative(zb)
            loss_f = torch.mean((dz_p - dz_true[idx]) ** 2)
            H_p = gen_map.hamiltonian(zb)
            loss_h = torch.mean((H_p - H_exact[idx]) ** 2)
            tot = loss_f + 2.0 * loss_h
            tot.backward()
            opt_gen.step()
        err_adv = system.compute_trajectory_error(gen_map)
        z_pred_adv = gen_map.integrate_symplectic_rk4(system.z0, t_dense).squeeze(1)
        H_pred_adv = gen_map.hamiltonian(z_pred_adv)
        drift_adv = (torch.abs(H_pred_adv - H0) / (torch.abs(H0) + 1e-6)).mean().item() * 100
        
    results["Theorem23_Advanced_Rel_L2"] = err_adv * 100
    results["Theorem23_Advanced_Drift"] = drift_adv
    
    return results

def run_four_way_master_suite(
    regime: str = "chaotic",
    epochs: int = 500,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    Executes full 4-Way Master Benchmark Suite across all 4 Celestial Systems.
    """
    dev = device if device is not None else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 85)
    print(f"  FOUR-WAY GRAND SYMPLECTIC CELESTIAL BENCHMARK SUITE")
    print(f"  Regime: {regime.upper()} | Epochs: {epochs} | Device: {dev}")
    print("=" * 85)
    
    systems = [
        BinaryQuasarHamiltonianSystem(regime=regime, device=dev),
        RestrictedSixBodyHamiltonianSystem(regime=regime, device=dev),
        SitnikovFiveBodyHamiltonianSystem(regime=regime, device=dev),
        MagneticYukawaHamiltonianSystem(regime=regime, device=dev),
    ]
    
    records = []
    for s in systems:
        print(f"\n--- Benchmarking System: {s.name} ({regime}) ---")
        t0 = time.time()
        res = run_four_way_system_benchmark(s, epochs=epochs, device=dev)
        el = time.time() - t0
        res["System"] = s.name
        res["Regime"] = regime
        res["Runtime_s"] = f"{el:.1f}s"
        records.append(res)
        print(f"  [+] Complete in {el:.1f}s | MLP: {res['Standard_MLP_Rel_L2']:.1f}% | Vanilla: {res['Vanilla_HNN_Rel_L2']:.1f}% | Thm1 (Sep): {res['Theorem1_Separable_Rel_L2']:.2f}% | Thm2/3 (Adv): {res['Theorem23_Advanced_Rel_L2']:.2f}%")
        
    df = pd.DataFrame(records)
    cols = ["System", "Regime", "Standard_MLP_Rel_L2", "Vanilla_HNN_Rel_L2", "Theorem1_Separable_Rel_L2", "Theorem23_Advanced_Rel_L2", "Theorem1_Separable_Drift", "Runtime_s"]
    df = df[cols]
    
    os.makedirs("results/data", exist_ok=True)
    out_csv = f"results/data/four_way_master_benchmarks_{regime}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[+] Saved 4-Way Master CSV: {out_csv}")
    
    print("\n" + "=" * 105)
    print("                   FOUR-WAY GRAND SCIENTIFIC BENCHMARK MATRIX")
    print("=" * 105)
    print(df.to_string(index=False))
    print("=" * 105)
    
    return df

if __name__ == "__main__":
    run_four_way_master_suite(regime="regular", epochs=10)
