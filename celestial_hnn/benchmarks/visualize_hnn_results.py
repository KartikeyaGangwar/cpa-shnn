import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from celestial_hnn.physics.binary_quasar import BinaryQuasarHamiltonianSystem
from celestial_hnn.physics.restricted_six_body import RestrictedSixBodyHamiltonianSystem
from celestial_hnn.physics.sitnikov_five_body import SitnikovFiveBodyHamiltonianSystem
from celestial_hnn.physics.magnetic_binary_yukawa import MagneticYukawaHamiltonianSystem
from celestial_hnn.training.hnn_trainer import CelestialHNNTrainer

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

def generate_triplet_comparison_figures(out_dir: str = "results/plots"):
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "="*70)
    print("  GENERATING 3-PANEL COMPARISON & CONVERGENCE VISUALIZATIONS")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    systems = [
        ("System 1: Binary Quasar", BinaryQuasarHamiltonianSystem(device=device), "system1_quasar"),
        ("System 2: Restricted 6-Body", RestrictedSixBodyHamiltonianSystem(device=device), "system2_sixbody"),
        ("System 3: Sitnikov 5-Body", SitnikovFiveBodyHamiltonianSystem(device=device), "system3_sitnikov"),
        ("System 4: Magnetic Yukawa", MagneticYukawaHamiltonianSystem(device=device), "system4_yukawa"),
    ]
    
    all_histories = {}
    
    for sys_title, sys_obj, file_prefix in systems:
        print(f"\n[+] Processing {sys_title}...")
        trainer = CelestialHNNTrainer(sys_obj, device=device)
        
        # Train baseline & HNN
        mlp_model, mlp_res = trainer.train_baseline_mlp(epochs=1200)
        hnn_model, hnn_res = trainer.train_hnn(epochs=1200)
        
        all_histories[sys_title] = {
            "mlp": mlp_res,
            "hnn": hnn_res
        }
        
        # Compute rollouts
        t_span = torch.linspace(0, sys_obj.T_max, 500, device=device)
        traj_true = sys_obj.ground_truth_trajectory(t_span).detach().cpu().numpy()
        
        z0 = sys_obj.z0.to(device)
        traj_mlp = mlp_model.integrate_symplectic_rk4(z0, t_span).squeeze(1).detach().cpu().numpy()
        traj_hnn = hnn_model.integrate_symplectic_rk4(z0, t_span).squeeze(1).detach().cpu().numpy()
        
        # -------------------------------------------------------------
        # 1. 3-PANEL COMPARISON PLOT: [Ground Truth | Standard MLP | Symplectic HNN]
        # -------------------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
        
        # Panel 1: Ground Truth
        ax1 = axes[0]
        ax1.plot(traj_true[:, 0], traj_true[:, 1], color="#1f77b4", linewidth=2.2, label="Ground Truth RK8(53)")
        ax1.scatter(traj_true[0, 0], traj_true[0, 1], color="green", s=80, marker="o", label=r"Start $z_0$", zorder=5)
        ax1.scatter(traj_true[-1, 0], traj_true[-1, 1], color="red", s=80, marker="x", label=r"End $z(T)$", zorder=5)
        ax1.set_title("(A) Ground Truth Trajectory\nExact Symplectic Invariant Flow", fontweight="bold")
        ax1.set_xlabel(r"$x$ / $q_1$")
        ax1.set_ylabel(r"$y$ / $q_2$")
        ax1.grid(True, linestyle="--")
        ax1.legend(loc="upper right")
        
        # Panel 2: Standard Baseline
        ax2 = axes[1]
        ax2.plot(traj_true[:, 0], traj_true[:, 1], color="gray", linestyle=":", linewidth=1.5, alpha=0.7, label="Exact Reference")
        ax2.plot(traj_mlp[:, 0], traj_mlp[:, 1], color="#d62728", linewidth=2.0, label=f"Standard Baseline\n(Rel L2: {mlp_res['rel_l2_error']*100:.2f}%)")
        ax2.scatter(traj_mlp[0, 0], traj_mlp[0, 1], color="green", s=80, marker="o", zorder=5)
        ax2.scatter(traj_mlp[-1, 0], traj_mlp[-1, 1], color="red", s=80, marker="x", zorder=5)
        ax2.set_title("(B) Standard Vector Field Baseline\nNon-Conservative Drift", fontweight="bold")
        ax2.set_xlabel(r"$x$ / $q_1$")
        ax2.set_ylabel(r"$y$ / $q_2$")
        ax2.grid(True, linestyle="--")
        ax2.legend(loc="upper right")
        
        # Panel 3: Symplectic HNN
        ax3 = axes[2]
        ax3.plot(traj_true[:, 0], traj_true[:, 1], color="gray", linestyle=":", linewidth=1.5, alpha=0.7, label="Exact Reference")
        ax3.plot(traj_hnn[:, 0], traj_hnn[:, 1], color="#2ca02c", linewidth=2.2, label=f"Symplectic HNN\n(Rel L2: {hnn_res['rel_l2_error']*100:.2f}%)")
        ax3.scatter(traj_hnn[0, 0], traj_hnn[0, 1], color="green", s=80, marker="o", zorder=5)
        ax3.scatter(traj_hnn[-1, 0], traj_hnn[-1, 1], color="red", s=80, marker="x", zorder=5)
        ax3.set_title(r"(C) Symplectic HNN (Ours)" + "\n" + r"$\dot{\mathbf{z}} = \mathbf{J}\nabla\mathcal{H}_\theta$ Exact Symplectic Invariant", fontweight="bold")
        ax3.set_xlabel(r"$x$ / $q_1$")
        ax3.set_ylabel(r"$y$ / $q_2$")
        ax3.grid(True, linestyle="--")
        ax3.legend(loc="upper right")
        
        fig.suptitle(f"{sys_title}: Phase-Space Trajectory Triplet Benchmark", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        plot_path = os.path.join(out_dir, f"{file_prefix}_triplet_comparison.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        print(f"  [+] Saved: {plot_path}")
        
    # -----------------------------------------------------------------
    # 2. MASTER CONVERGENCE CURVES (Loss vs Epochs for All 4 Systems)
    # -----------------------------------------------------------------
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
        ax.semilogy(epochs_hnn, loss_hnn_field, color="#1f77b4", linewidth=2.2, label="HNN Symplectic Field Loss")
        ax.semilogy(epochs_hnn, loss_hnn_energy, color="#2ca02c", linestyle="-.", linewidth=1.8, label="HNN Symplectic Energy Loss")
        
        ax.set_title(sys_title, fontweight="bold")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss (Log Scale)")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(loc="upper right")
        
    fig.suptitle("Symplectic HNN vs Standard Vector Field Baseline Convergence Curves", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    conv_path = os.path.join(out_dir, "master_convergence_curves.png")
    plt.savefig(conv_path, bbox_inches="tight")
    plt.close()
    print(f"\n[+] Master Convergence Curves saved: {conv_path}")

if __name__ == "__main__":
    generate_triplet_comparison_figures()
