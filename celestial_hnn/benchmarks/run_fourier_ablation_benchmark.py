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

from celestial_hnn.models.hnn import HamiltonianNeuralNetwork
from celestial_hnn.models.structured_separable_hnn import StructuredSeparableHNN
from celestial_hnn.models.separable_generating_hnn import SeparableGeneratingMapHNN
from celestial_hnn.models.grand_unified_engine import GrandUnifiedSymplecticEngine
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
    Conducts a rigorous ablation comparing Fourier Positional Encodings vs Pure Smooth C^infinity MLPs
    across representative Autonomous and Non-Autonomous Celestial Systems.
    """
    dev = device if device is not None else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 125)
    print(f"  FOURIER POSITIONAL ENCODING ABLATION STUDY (FOURIER VS NON-FOURIER)")
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
        
        # Test Pairs: [No-Fourier] vs [With-Fourier]
        if not is_non_auto:
            models_to_test = {
                "Thm1_NoFourier": StructuredSeparableHNN(spatial_dim=s.spatial_dim, n_coriolis=n_c, hidden_dim=256, use_fourier=False).to(dev),
                "Thm1_WithFourier": StructuredSeparableHNN(spatial_dim=s.spatial_dim, n_coriolis=n_c, hidden_dim=256, use_fourier=True).to(dev),
                "Combo13_NoFourier": SeparableGeneratingMapHNN(spatial_dim=s.spatial_dim, n_coriolis=n_c, hidden_dim=256, use_fourier=False).to(dev),
                "Combo13_WithFourier": SeparableGeneratingMapHNN(spatial_dim=s.spatial_dim, n_coriolis=n_c, hidden_dim=256, use_fourier=True).to(dev),
            }
            row = {"System": s.name, "Regime": regime}
            for name, m in models_to_test.items():
                res = train_model_with_cpa_time_marching(m, s, n_windows=n_windows, epochs_per_window=ep_win, use_lbfgs=True, lbfgs_max_iter=lbfgs_max_iter)
                row[name] = res["rel_l2_error"]
                print(f"  --> [{name}]: Error = {res['rel_l2_error']:.2f}%")
            records.append(row)
        else:
            models_to_test = {
                "GrandUnified_NoFourier": GrandUnifiedSymplecticEngine(spatial_dim=s.spatial_dim, n_coriolis=n_c, hidden_dim=256, use_fourier=False).to(dev),
                "GrandUnified_WithFourier": GrandUnifiedSymplecticEngine(spatial_dim=s.spatial_dim, n_coriolis=n_c, hidden_dim=256, use_fourier=True).to(dev),
            }
            row = {"System": s.name, "Regime": regime}
            for name, m in models_to_test.items():
                res = train_non_autonomous_model(m, s, n_windows=n_windows, epochs_per_window=ep_win, use_lbfgs=True, lbfgs_max_iter=lbfgs_max_iter)
                row[name] = res["rel_l2_error"]
                print(f"  --> [{name}]: Error = {res['rel_l2_error']:.2f}%")
            records.append(row)
            
    df = pd.DataFrame(records)
    os.makedirs("results/data", exist_ok=True)
    out_csv = f"results/data/fourier_ablation_benchmarks_{regime}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[+] Saved Fourier Ablation CSV: {out_csv}")
    
    print("\n" + "=" * 125)
    print("                 FOURIER ABLATION STUDY COMPARATIVE MATRIX")
    print("=" * 125)
    print(df.to_string(index=False))
    print("=" * 125)
    return df

if __name__ == "__main__":
    run_fourier_ablation_study(regime="regular", epochs=10)
