import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import torch
from celestial_pinn.physics.restricted_six_body import RestrictedSixBodySquareSystem
from celestial_pinn.training.two_stage_trainer import TwoStageCelestialASPINNTrainer
from celestial_pinn.training.baseline_trainers import BaselineCelestialTrainer

def run_system2_benchmark(device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "#"*70)
    print("  EXECUTING SYSTEM II BENCHMARK: RESTRICTED SIX-BODY SQUARE SYSTEM")
    print("#"*70)
    
    system = RestrictedSixBodySquareSystem(device=device)
    
    print("\n--- Running Standard PINN Baseline ---")
    base_trainer = BaselineCelestialTrainer(system, device=device)
    std_model, std_res = base_trainer.train_standard_pinn(epochs=600)
    
    print("\n--- Running AS-PINN Framework ---")
    as_trainer = TwoStageCelestialASPINNTrainer(system, device=device)
    disc_info = as_trainer.run_stage1_discovery(min_epochs=150, max_epochs=400, max_subspaces=16)
    as_model, as_res = as_trainer.train_stage2_production(
        discovered_info=disc_info,
        adamw_epochs=600,
        lbfgs_steps=150
    )
    
    results = {
        "system": "RestrictedSixBodySquareSystem",
        "standard_pinn": {
            "final_loss": std_res["final_loss_total"],
            "rel_l2_error": std_res["final_rel_l2_err"],
            "time_sec": std_res["wall_time"],
        },
        "as_pinn": {
            "final_loss": as_res["final_loss_total"],
            "rel_l2_error": as_res["final_rel_l2_err"],
            "discovered_N": as_res["num_subspaces"],
            "time_sec": as_res["wall_time"],
        },
    }
    
    os.makedirs("results/data", exist_ok=True)
    with open("results/data/system2_sixbody_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n[+] System II Benchmark Complete! AS-PINN Rel L2: {as_res['final_rel_l2_err']*100:.3f}% vs Std PINN: {std_res['final_rel_l2_err']*100:.3f}%")
    return results

if __name__ == "__main__":
    run_system2_benchmark()
