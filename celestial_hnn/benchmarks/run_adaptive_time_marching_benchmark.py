import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json, torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from celestial_hnn.physics.binary_quasar import BinaryQuasarHamiltonianSystem
from celestial_hnn.physics.restricted_six_body import RestrictedSixBodyHamiltonianSystem
from celestial_hnn.physics.sitnikov_five_body import SitnikovFiveBodyHamiltonianSystem
from celestial_hnn.physics.magnetic_binary_yukawa import MagneticYukawaHamiltonianSystem
from celestial_hnn.training.hnn_trainer import CelestialHNNTrainer
from celestial_hnn.training.adaptive_time_marching_trainer import AdaptiveTimeMarchingHNNTrainer

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "axes.linewidth": 1.0,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.35,
})

def run_adaptive_time_marching_master_suite(
    regime: str = "chaotic",
    epochs_per_window: int = 500,
    n_windows: int = 6,
    use_lbfgs: bool = True,
    save_plots: bool = True,
    out_data_dir: str = "results/data",
    out_plot_dir: str = "results/plots",
):
    os.makedirs(out_data_dir, exist_ok=True)
    os.makedirs(out_plot_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*85)
    print("  ADAPTIVE ENERGY-GUIDED TIME-MARCHING SYMPLECTIC HNN BENCHMARK SUITE")
    print(f"  Regime: {regime.upper()} | Windows: {n_windows} | Device: {device} | Torch: {torch.__version__}")
    print("="*85)
    
    systems = [
        ("System 1: Binary Quasar", BinaryQuasarHamiltonianSystem(regime=regime, device=device), f"atm_{regime}_system1_quasar"),
        ("System 2: Restricted 6-Body", RestrictedSixBodyHamiltonianSystem(regime=regime, device=device), f"atm_{regime}_system2_sixbody"),
        ("System 3: Sitnikov 5-Body", SitnikovFiveBodyHamiltonianSystem(regime=regime, device=device), f"atm_{regime}_system3_sitnikov"),
        ("System 4: Magnetic Yukawa", MagneticYukawaHamiltonianSystem(regime=regime, device=device), f"atm_{regime}_system4_yukawa"),
    ]
    
    results_summary = []
    all_histories = {}
    
    for sys_title, s, file_prefix in systems:
        print(f"\n{'#'*75}")
        print(f"  BENCHMARKING: {sys_title.upper()} (Regime: {regime.upper()})")
        print(f"{'#'*75}")
        
        # 1. Train Standard Baseline MLP
        print("\n--- Training Standard Baseline Vector Field MLP ---")
        base_trainer = CelestialHNNTrainer(s, device=device)
        mlp_model, mlp_res = base_trainer.train_baseline_mlp(epochs=epochs_per_window * n_windows)
        
        # 2. Train Adaptive Time-Marching HNN
        print("\n--- Training Causality-Preserving Adaptive Time-Marching HNN ---")
        atm_trainer = AdaptiveTimeMarchingHNNTrainer(
            system=s,
            n_windows=n_windows,
            hidden_dim=256,
            layers=4,
            device=device,
        )
        
        hnn_model, hnn_res = atm_trainer.train_adaptive_time_marching(
            epochs_per_window=epochs_per_window,
            use_lbfgs=use_lbfgs,
            lbfgs_max_iter=60,
            verbose=True,
        )
        
        improvement_l2 = (mlp_res["rel_l2_error"] - hnn_res["rel_l2_error"]) / max(mlp_res["rel_l2_error"], 1e-8) * 100.0
        
        # Save individual system JSON
        sys_json_path = os.path.join(out_data_dir, f"{file_prefix}_results.json")
        full_sys_data = {
            "title": sys_title,
            "system_name": s.name,
            "regime": regime,
            "n_windows": n_windows,
            "standard_mlp": mlp_res,
            "adaptive_time_marching_hnn": hnn_res,
            "trajectory_accuracy_gain_pct": improvement_l2,
        }
        with open(sys_json_path, "w", encoding="utf-8") as f:
            json.dump(full_sys_data, f, indent=2)
        print(f"  [+] Saved System JSON: {sys_json_path}")
        
        results_summary.append({
            "Benchmark System": s.name,
            "Regime": regime,
            "Windows": n_windows,
            "Standard MLP Rel L2": f"{mlp_res['rel_l2_error']*100:.2f}%",
            "Adaptive HNN Rel L2": f"{hnn_res['rel_l2_error']*100:.3f}%",
            "Accuracy Gain": f"{improvement_l2:+.1f}%",
            "Energy Drift": f"{hnn_res['energy_drift_rel']*100:.4f}%",
            "Runtime (s)": f"{hnn_res['training_time_seconds']:.1f}s",
        })
        
        all_histories[sys_title] = {
            "mlp": mlp_res,
            "hnn": hnn_res,
        }
        
        # 3. Generate 3-Panel Comparison Trajectory Plot
        if save_plots:
            t_span = torch.linspace(0, s.T_max, 600, device=device)
            traj_true = s.ground_truth_trajectory(t_span).detach().cpu().numpy()
            z0 = s.z0.to(device)
            traj_mlp = mlp_model.integrate_symplectic_rk4(z0, t_span).squeeze(1).detach().cpu().numpy()
            traj_hnn = hnn_model.integrate_symplectic_rk4(z0, t_span).squeeze(1).detach().cpu().numpy()
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
            
            # Panel A: Ground Truth
            ax1 = axes[0]
            ax1.plot(traj_true[:, 0], traj_true[:, 1], color="#1f77b4", linewidth=2.2, label="DOP853 Reference Flow")
            ax1.scatter(traj_true[0, 0], traj_true[0, 1], color="green", s=80, marker="o", label=r"Start $z_0$", zorder=5)
            ax1.scatter(traj_true[-1, 0], traj_true[-1, 1], color="red", s=80, marker="x", label=r"End $z(T)$", zorder=5)
            ax1.set_title(f"(A) Ground Truth Trajectory\nExact Symplectic Invariant Flow", fontweight="bold")
            ax1.set_xlabel(r"$x$ / $q_1$" if s.spatial_dim == 2 else r"$z$ (Position)")
            ax1.set_ylabel(r"$y$ / $q_2$" if s.spatial_dim == 2 else r"$p_z$ (Momentum)")
            ax1.grid(True, linestyle="--")
            ax1.legend(loc="upper right")
            
            # Panel B: Standard Baseline
            ax2 = axes[1]
            ax2.plot(traj_true[:, 0], traj_true[:, 1], color="gray", linestyle=":", linewidth=1.5, alpha=0.7, label="Exact Reference")
            ax2.plot(traj_mlp[:, 0], traj_mlp[:, 1], color="#d62728", linewidth=2.0, label=f"Standard Baseline\n(Rel L2: {mlp_res['rel_l2_error']*100:.2f}%)")
            ax2.scatter(traj_mlp[0, 0], traj_mlp[0, 1], color="green", s=80, marker="o", zorder=5)
            ax2.scatter(traj_mlp[-1, 0], traj_mlp[-1, 1], color="red", s=80, marker="x", zorder=5)
            ax2.set_title(f"(B) Standard Vector Field Baseline\nNon-Conservative Dissipation", fontweight="bold")
            ax2.set_xlabel(r"$x$ / $q_1$" if s.spatial_dim == 2 else r"$z$ (Position)")
            ax2.set_ylabel(r"$y$ / $q_2$" if s.spatial_dim == 2 else r"$p_z$ (Momentum)")
            ax2.grid(True, linestyle="--")
            ax2.legend(loc="upper right")
            
            # Panel C: Adaptive Time-Marching HNN
            ax3 = axes[2]
            ax3.plot(traj_true[:, 0], traj_true[:, 1], color="gray", linestyle=":", linewidth=1.5, alpha=0.7, label="Exact Reference")
            ax3.plot(traj_hnn[:, 0], traj_hnn[:, 1], color="#2ca02c", linewidth=2.2, label=f"Adaptive Time-Marching HNN\n(Rel L2: {hnn_res['rel_l2_error']*100:.2f}%)")
            ax3.scatter(traj_hnn[0, 0], traj_hnn[0, 1], color="green", s=80, marker="o", zorder=5)
            ax3.scatter(traj_hnn[-1, 0], traj_hnn[-1, 1], color="red", s=80, marker="x", zorder=5)
            ax3.set_title(r"(C) Adaptive Time-Marching HNN (Ours)" + f"\nCausality-Preserving Symplectic Flow", fontweight="bold")
            ax3.set_xlabel(r"$x$ / $q_1$" if s.spatial_dim == 2 else r"$z$ (Position)")
            ax3.set_ylabel(r"$y$ / $q_2$" if s.spatial_dim == 2 else r"$p_z$ (Momentum)")
            ax3.grid(True, linestyle="--")
            ax3.legend(loc="upper right")
            
            fig.suptitle(f"{sys_title} - Adaptive Time-Marching Triplet Benchmark ({regime.title()})", fontsize=15, fontweight="bold", y=1.02)
            plt.tight_layout()
            plot_path = os.path.join(out_plot_dir, f"{file_prefix}_triplet_comparison.png")
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            print(f"  [+] Saved Triplet Plot: {plot_path}")

    # 4. Save Master Summary CSV
    df = pd.DataFrame(results_summary)
    csv_path = os.path.join(out_data_dir, f"adaptive_time_marching_benchmarks_{regime}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[+] Master CSV Saved: {csv_path}")
    
    # 5. Master Multi-System Convergence Figure
    if save_plots:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
        axes = axes.flatten()
        for idx, (sys_title, _, file_prefix) in enumerate(systems):
            ax = axes[idx]
            hist = all_histories[sys_title]
            
            epochs_mlp = hist["mlp"]["epochs_logged"]
            loss_mlp = hist["mlp"]["loss_history"]
            
            epochs_hnn = hist["hnn"]["epochs_logged"]
            loss_hnn_field = hist["hnn"]["field_loss_history"]
            loss_hnn_energy = hist["hnn"]["energy_loss_history"]
            
            ax.semilogy(epochs_mlp, loss_mlp, color="#d62728", linestyle="--", linewidth=2.0, label="Standard Baseline Loss")
            ax.semilogy(epochs_hnn, loss_hnn_field, color="#1f77b4", linewidth=2.2, label="Adaptive HNN Field Loss")
            ax.semilogy(epochs_hnn, loss_hnn_energy, color="#2ca02c", linestyle="-.", linewidth=1.8, label="Adaptive HNN Energy Loss")
            
            ax.set_title(sys_title, fontweight="bold")
            ax.set_xlabel("Logged Iterations")
            ax.set_ylabel("Loss (Log Scale)")
            ax.grid(True, which="both", linestyle="--", alpha=0.4)
            ax.legend(loc="upper right")
            
        fig.suptitle(f"Adaptive Time-Marching Convergence Curves ({regime.title()} Regime)", fontsize=16, fontweight="bold", y=1.01)
        plt.tight_layout()
        conv_path = os.path.join(out_plot_dir, f"adaptive_time_marching_master_convergence_{regime}.png")
        plt.savefig(conv_path, bbox_inches="tight")
        plt.close()
        print(f"[+] Master Convergence Plot Saved: {conv_path}")
        
    print("\n" + "="*95)
    print(f"         ADAPTIVE TIME-MARCHING CELESTIAL QUANTITATIVE MATRIX ({regime.upper()})")
    print("="*95)
    print(df.to_string(index=False))
    print("="*95 + "\n")
    return df

if __name__ == "__main__":
    run_adaptive_time_marching_master_suite(regime="chaotic", epochs_per_window=500, n_windows=6)
