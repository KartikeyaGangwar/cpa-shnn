import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json, torch
from celestial_hnn.physics.sitnikov_five_body import SitnikovFiveBodyHamiltonianSystem
from celestial_hnn.training.hnn_trainer import CelestialHNNTrainer

def run_system3_benchmark(device=None):
    print("\n" + "#"*70)
    print("  SYSTEM 3 BENCHMARK: SITNIKOV FIVE-BODY HAMILTONIAN DYNAMICS")
    print("#"*70)
    system = SitnikovFiveBodyHamiltonianSystem(device=device)
    trainer = CelestialHNNTrainer(system, device=device)
    
    mlp_model, mlp_res = trainer.train_baseline_mlp(epochs=1200)
    hnn_model, hnn_res = trainer.train_hnn(epochs=1200)
    
    results = {
        "system": "SitnikovFiveBodyHamiltonianSystem",
        "standard_mlp": mlp_res,
        "hnn_symplectic": hnn_res,
    }
    os.makedirs("results/data", exist_ok=True)
    with open("results/data/system3_sitnikov_hnn.json", "w") as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    run_system3_benchmark()
