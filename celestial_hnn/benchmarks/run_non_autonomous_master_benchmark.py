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
    model: nn.Module,
    system: Any,
    n_windows: int = 5,
    epochs_per_window: int = 80,
    use_lbfgs: bool = True,
    lbfgs_max_iter: int = 50
) -> Dict[str, Any]:
    dev = system.device
    T_max = system.T_max
    window_boundaries = torch.linspace(0, T_max, n_windows + 1, device=dev)
    
    for w in range(n_windows):
        t_curr_end = window_boundaries[w + 1]
        t_w = torch.linspace(0, t_curr_end, 1000 * (w + 1), device=dev)
        z_w = system.ground_truth_trajectory(t_w)
        dz_w = system.canonical_derivatives(z_w)
        
        is_autonomous_arch = (getattr(model, "state_dim", 2 * system.spatial_dim) == 2 * system.spatial_dim)
        if is_autonomous_arch:
            if system.spatial_dim == 1:
                zb_train = torch.cat([z_w[:, 0:1], z_w[:, 2:3]], dim=-1)
                dzb_train = torch.cat([dz_w[:, 0:1], dz_w[:, 2:3]], dim=-1)
            else:
                zb_train = torch.cat([z_w[:, :2], z_w[:, 3:5]], dim=-1)
                dzb_train = torch.cat([dz_w[:, :2], dz_w[:, 3:5]], dim=-1)
        else:
            zb_train = z_w
            dzb_train = dz_w
            
        opt_adam = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_adam, T_max=epochs_per_window, eta_min=1e-5)
        
        for _ in range(epochs_per_window):
            opt_adam.zero_grad()
            idx = torch.randint(0, len(zb_train), (min(1024, len(zb_train)),), device=dev)
            zb = zb_train[idx]
            zt = zb + torch.randn_like(zb) * 0.025
            za = torch.cat([zb, zt], dim=0)
            
            if is_autonomous_arch:
                if system.spatial_dim == 1:
                    dza_deriv = system.canonical_derivatives(torch.cat([zt[:, 0:1], torch.zeros_like(zt[:, 0:1]), zt[:, 1:2], torch.zeros_like(zt[:, 1:2])], dim=-1))
                    dza = torch.cat([dzb_train[idx], torch.cat([dza_deriv[:, 0:1], dza_deriv[:, 2:3]], dim=-1)], dim=0)
                else:
                    dza_deriv = system.canonical_derivatives(torch.cat([zt[:, :2], torch.zeros_like(zt[:, 0:1]), zt[:, 2:], torch.zeros_like(zt[:, 0:1])], dim=-1))
                    dza = torch.cat([dzb_train[idx], torch.cat([dza_deriv[:, :2], dza_deriv[:, 3:5]], dim=-1)], dim=0)
            else:
                dza = torch.cat([dzb_train[idx], system.canonical_derivatives(zt)], dim=0)
                
            dz_p = model.time_derivative(za) if hasattr(model, "time_derivative") else model(za)
            loss = torch.mean((dz_p - dza) ** 2)
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
            zb_l = zb_train
            dza_l = dzb_train
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
                
    t_dense = torch.linspace(0, T_max, 2500, device=dev)
    z_gt = system.ground_truth_trajectory(t_dense)
    is_autonomous_arch = (getattr(model, "state_dim", 2 * system.spatial_dim) == 2 * system.spatial_dim)
    
    if is_autonomous_arch:
        if system.spatial_dim == 1:
            z0_input = torch.tensor([system.z0_val, system.pz0_val], dtype=torch.float32, device=dev)
            z_pred_raw = model.integrate_symplectic_rk4(z0_input, t_dense).squeeze(1)
            num = torch.norm(z_pred_raw[:, 0] - z_gt[:, 0])
            den = torch.norm(z_gt[:, 0]) + 1e-7
            err = (num / den).item() * 100
            z_pred = torch.cat([z_pred_raw[:, 0:1], t_dense.unsqueeze(-1), z_pred_raw[:, 1:2], torch.zeros_like(z_pred_raw[:, 0:1])], dim=-1)
        else:
            z0_input = torch.tensor([system.x0_val, system.y0_val, system.px0_val, system.py0_val], dtype=torch.float32, device=dev)
            z_pred_raw = model.integrate_symplectic_rk4(z0_input, t_dense).squeeze(1)
            num = torch.norm(z_pred_raw[:, :2] - z_gt[:, :2])
            den = torch.norm(z_gt[:, :2]) + 1e-7
            err = (num / den).item() * 100
            z_pred = torch.cat([z_pred_raw[:, :2], t_dense.unsqueeze(-1), z_pred_raw[:, 2:], torch.zeros_like(z_pred_raw[:, 0:1])], dim=-1)
    else:
        err = system.compute_trajectory_error(model) * 100
        z_pred = model.integrate_symplectic_rk4(system.z0, t_dense).squeeze(1)
        
    return {"rel_l2_error": err, "z_pred": z_pred}

def run_non_autonomous_master_suite(
    regime: str = "chaotic",
    epochs: int = 400,
    n_windows: int = 5,
    lbfgs_max_iter: int = 50,
    device: Optional[torch.device] = None
) -> pd.DataFrame:
    dev = device if device is not None else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 125)
    print(f"  NON-AUTONOMOUS CELESTIAL MASTER BENCHMARK SUITE")
    print(f"  Regime: {regime.upper()} | Epochs: {epochs} | Windows: {n_windows} | L-BFGS Max Iter: {lbfgs_max_iter} | Device: {dev}")
    print("=" * 125)
    
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
        
        # 5 Clean Non-Autonomous Models (Theorems 1 & 2 only)
        models = {
            "1_Standard_PINN_MLP": BaselineVectorFieldMLP(state_dim=s.state_dim, hidden_dim=256).to(dev),
            "2_Vanilla_HNN_2019": HamiltonianNeuralNetwork(spatial_dim=s.spatial_dim, hidden_dim=256, use_fourier=True).to(dev),
            "3_CPA_SHNN_Core": StructuredSeparableHNN(spatial_dim=s.spatial_dim, n_coriolis=n_c, hidden_dim=256, use_fourier=True).to(dev),
            "4_Theorem2_ExtendedContactHNN": ExtendedPhaseSpaceHNN(spatial_dim=s.spatial_dim, hidden_dim=256, use_fourier=True).to(dev),
            "5_Theorem1_plus_2_SeparableExtendedHNN": SeparableExtendedContactHNN(spatial_dim=s.spatial_dim, n_coriolis=n_c, hidden_dim=256, use_fourier=True).to(dev),
        }
        
        res_dict = {}
        preds_dict = {}
        for name, m in models.items():
            res = train_non_autonomous_model(
                m, s, n_windows=n_windows, epochs_per_window=ep_win, use_lbfgs=True, lbfgs_max_iter=lbfgs_max_iter
            )
            res_dict[name] = res["rel_l2_error"]
            preds_dict[name] = res["z_pred"]
            print(f"  --> Model [{name}]: Error = {res['rel_l2_error']:.2f}%")
            
        el = time.time() - t0
        res_dict["System"] = s.name
        res_dict["Regime"] = regime
        res_dict["Runtime_s"] = f"{el:.1f}s"
        records.append(res_dict)
        
    df = pd.DataFrame(records)
    cols = ["System", "Regime", "1_Standard_PINN_MLP", "2_Vanilla_HNN_2019", "3_CPA_SHNN_Core", "4_Theorem2_ExtendedContactHNN", "5_Theorem1_plus_2_SeparableExtendedHNN", "Runtime_s"]
    df = df[cols]
    
    os.makedirs("results/data", exist_ok=True)
    out_csv = f"results/data/non_autonomous_master_benchmarks_{regime}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[+] Saved Non-Autonomous Master CSV: {out_csv}")
    
    print("\n" + "=" * 125)
    print("                 NON-AUTONOMOUS CELESTIAL MASTER BENCHMARK MATRIX")
    print("=" * 125)
    print(df.to_string(index=False))
    print("=" * 125)
    return df

if __name__ == "__main__":
    run_non_autonomous_master_suite(regime="regular", epochs=10)
