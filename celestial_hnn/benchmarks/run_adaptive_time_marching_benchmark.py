import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json, torch
import pandas as pd

from celestial_hnn.physics.binary_quasar import BinaryQuasarHamiltonianSystem
from celestial_hnn.physics.restricted_six_body import RestrictedSixBodyHamiltonianSystem
from celestial_hnn.physics.sitnikov_five_body import SitnikovFiveBodyHamiltonianSystem
from celestial_hnn.physics.magnetic_binary_yukawa import MagneticYukawaHamiltonianSystem
from celestial_hnn.training.adaptive_time_marching_trainer import AdaptiveTimeMarchingHNNTrainer

def run_adaptive_time_marching_master_suite(
    regime: str = "chaotic",
    epochs_per_window: int = 300,
    n_windows: int = 4,
    use_lbfgs: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*85)
    print("  ADAPTIVE ENERGY-GUIDED TIME-MARCHING SYMPLECTIC HNN BENCHMARK SUITE")
    print(f"  Regime: {regime.upper()} | Windows: {n_windows} | Device: {device} | Torch: {torch.__version__}")
    print("="*85)
    
    systems = [
        BinaryQuasarHamiltonianSystem(regime=regime, device=device),
        RestrictedSixBodyHamiltonianSystem(regime=regime, device=device),
        SitnikovFiveBodyHamiltonianSystem(regime=regime, device=device),
        MagneticYukawaHamiltonianSystem(regime=regime, device=device),
    ]
    
    results_summary = []
    
    for s in systems:
        print(f"\n{'#'*70}")
        print(f"  BENCHMARKING: {s.name.upper()} (Regime: {regime.upper()})")
        print(f"{'#'*70}")
        
        trainer = AdaptiveTimeMarchingHNNTrainer(
            system=s,
            n_windows=n_windows,
            hidden_dim=256,
            layers=4,
            device=device,
        )
        
        model, res = trainer.train_adaptive_time_marching(
            epochs_per_window=epochs_per_window,
            use_lbfgs=use_lbfgs,
            lbfgs_max_iter=30,
            verbose=True,
        )
        
        results_summary.append({
            "Benchmark System": s.name,
            "Regime": regime,
            "Windows": n_windows,
            "Symplectic Field Loss": f"{res['final_field_loss']:.3e}",
            "Energy Loss": f"{res['final_energy_loss']:.3e}",
            "Trajectory Rel L2 Error": f"{res['rel_l2_error']*100:.3f}%",
            "Energy Invariant Drift": f"{res['energy_drift_rel']*100:.4f}%",
            "Runtime (s)": f"{res['training_time_seconds']:.1f}s",
        })
        
    df = pd.DataFrame(results_summary)
    os.makedirs("results/data", exist_ok=True)
    df.to_csv(f"results/data/adaptive_time_marching_benchmarks_{regime}.csv", index=False)
    
    print("\n" + "="*90)
    print(f"      ADAPTIVE TIME-MARCHING CELESTIAL QUANTITATIVE MATRIX ({regime.upper()})")
    print("="*90)
    print(df.to_string(index=False))
    print("="*90 + "\n")
    return df

if __name__ == "__main__":
    run_adaptive_time_marching_master_suite(regime="chaotic", epochs_per_window=300, n_windows=4)
