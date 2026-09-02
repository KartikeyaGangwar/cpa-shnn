import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import torch
import pandas as pd
from celestial_pinn.benchmarks.run_system1_quasar import run_system1_benchmark
from celestial_pinn.benchmarks.run_system2_sixbody import run_system2_benchmark
from celestial_pinn.benchmarks.run_system3_sitnikov import run_system3_benchmark
from celestial_pinn.benchmarks.run_system4_yukawa import run_system4_benchmark

def run_master_celestial_suite():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*80)
    print("  AUTONOMOUS ADAPTIVE N-SUBSPACE PINN (AS-PINN) CELESTIAL MASTER BENCHMARK")
    print(f"  Execution Device: {device} | Torch Version: {torch.__version__}")
    print("="*80)
    
    r1 = run_system1_benchmark(device)
    r2 = run_system2_benchmark(device)
    r3 = run_system3_benchmark(device)
    r4 = run_system4_benchmark(device)
    
    all_results = [r1, r2, r3, r4]
    
    summary_rows = []
    for r in all_results:
        sys_name = r["system"]
        std_loss = r["standard_pinn"]["final_loss"]
        std_l2 = r["standard_pinn"]["rel_l2_error"]
        as_loss = r["as_pinn"]["final_loss"]
        as_l2 = r["as_pinn"]["rel_l2_error"]
        n_sub = r["as_pinn"]["discovered_N"]
        
        improvement_loss = std_loss / max(as_loss, 1e-12)
        improvement_l2 = (std_l2 - as_l2) / max(std_l2, 1e-8) * 100.0
        
        summary_rows.append({
            "Benchmark System": sys_name,
            "Discovered N*": n_sub,
            "Standard PINN Loss": f"{std_loss:.3e}",
            "AS-PINN Loss": f"{as_loss:.3e}",
            "Loss Reduction": f"{improvement_loss:.1f}x",
            "Standard PINN L2": f"{std_l2*100:.2f}%",
            "AS-PINN L2": f"{as_l2*100:.2f}%",
            "Accuracy Gain": f"{improvement_l2:+.1f}%",
        })
        
    df = pd.DataFrame(summary_rows)
    os.makedirs("results/data", exist_ok=True)
    df.to_csv("results/data/master_celestial_benchmarks.csv", index=False)
    
    print("\n" + "="*85)
    print("                 MASTER CELESTIAL QUANTITATIVE RESULTS MATRIX")
    print("="*85)
    print(df.to_string(index=False))
    print("="*85 + "\n")

if __name__ == "__main__":
    run_master_celestial_suite()
