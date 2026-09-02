import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json, torch
from celestial_hnn.physics.magnetic_binary_yukawa import MagneticYukawaHamiltonianSystem
from celestial_hnn.training.hnn_trainer import CelestialHNNTrainer

def run_system4_benchmark(device=None, regime="regular"):
    print("\n" + "#"*70)
    print(f"  SYSTEM 4 BENCHMARK: MAGNETIC YUKAWA BINARY HAMILTONIAN DYNAMICS ({regime.upper()})")
    print("#"*70)
    system = MagneticYukawaHamiltonianSystem(regime=regime, device=device)
    trainer = CelestialHNNTrainer(system, device=device)
    
    mlp_model, mlp_res = trainer.train_baseline_mlp(epochs=1200)
    hnn_model, hnn_res = trainer.train_hnn(epochs=1200)
    
    results = {
        "system": f"MagneticYukawa_{regime}",
        "regime": regime,
        "standard_mlp": mlp_res,
        "hnn_symplectic": hnn_res,
    }
    os.makedirs("results/data", exist_ok=True)
    with open(f"results/data/system4_yukawa_{regime}_hnn.json", "w") as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    run_system4_benchmark()
