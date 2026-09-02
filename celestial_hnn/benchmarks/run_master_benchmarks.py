import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json, torch
import pandas as pd
from celestial_hnn.benchmarks.run_system1_quasar import run_system1_benchmark
from celestial_hnn.benchmarks.run_system2_sixbody import run_system2_benchmark
from celestial_hnn.benchmarks.run_system3_sitnikov import run_system3_benchmark
from celestial_hnn.benchmarks.run_system4_yukawa import run_system4_benchmark

def run_master_celestial_hnn_suite(regime="regular"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*80)
    print(f"  SYMPLECTIC HNN CELESTIAL MASTER BENCHMARK (REGIME: {regime.upper()})")
    print(f"  Execution Device: {device} | Torch Version: {torch.__version__}")
    print("="*80)
    
    r1 = run_system1_benchmark(device, regime=regime)
    r2 = run_system2_benchmark(device, regime=regime)
    r3 = run_system3_benchmark(device, regime=regime)
    r4 = run_system4_benchmark(device, regime=regime)
    
    all_results = [r1, r2, r3, r4]
    
    summary_rows = []
    for r in all_results:
        sys_name = r["system"]
        std_loss = r["standard_mlp"]["final_loss"]
        std_l2 = r["standard_mlp"]["rel_l2_error"]
        hnn_loss = r["hnn_symplectic"]["final_loss"]
        hnn_l2 = r["hnn_symplectic"]["rel_l2_error"]
        
        improvement_l2 = (std_l2 - hnn_l2) / max(std_l2, 1e-8) * 100.0
        
        summary_rows.append({
            "Benchmark System": sys_name,
            "Standard MLP Loss": f"{std_loss:.3e}",
            "HNN Symplectic Loss": f"{hnn_loss:.3e}",
            "Standard MLP Rel L2": f"{std_l2*100:.2f}%",
            "HNN Symplectic Rel L2": f"{hnn_l2*100:.3f}%",
            "Trajectory Accuracy Gain": f"{improvement_l2:+.1f}%",
        })
        
    df = pd.DataFrame(summary_rows)
    os.makedirs("results/data", exist_ok=True)
    df.to_csv(f"results/data/master_celestial_hnn_benchmarks_{regime}.csv", index=False)
    
    print("\n" + "="*85)
    print(f"        MASTER CELESTIAL HAMILTONIAN MATRIX ({regime.upper()})")
    print("="*85)
    print(df.to_string(index=False))
    print("="*85 + "\n")
    return df

if __name__ == "__main__":
    run_master_celestial_hnn_suite("regular")
