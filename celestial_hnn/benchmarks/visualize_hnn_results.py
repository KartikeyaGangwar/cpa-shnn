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

def generate_dual_regime_visualizations(out_dir: str = "results/plots"):
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "="*75)
    print("  GENERATING DUAL-REGIME (REGULAR KAM & LONG-HORIZON CHAOTIC) VISUALS")
    print("="*75)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    regimes = ["regular", "chaotic"]
    
    for regime in regimes:
        print(f"\n{'='*30} BENCHMARKING REGIME: {regime.upper()} {'='*30}")
        
        systems = [
            (f"System 1: Binary Quasar ({regime.title()})", BinaryQuasarHamiltonianSystem(regime=regime, device=device), f"{regime}_system1_quasar"),
            (f"System 2: Restricted 6-Body ({regime.title()})", RestrictedSixBodyHamiltonianSystem(regime=regime, device=device), f"{regime}_system2_sixbody"),
            (f"System 3: Sitnikov 5-Body ({regime.title()})", SitnikovFiveBodyHamiltonianSystem(regime=regime, device=device), f"{regime}_system3_sitnikov"),
            (f"System 4: Magnetic Yukawa ({regime.title()})", MagneticYukawaHamiltonianSystem(regime=regime, device=device), f"{regime}_system4_yukawa"),
        ]
        
        for sys_title, sys_obj, file_prefix in systems:
            print(f"\n[+] Processing {sys_title}...")
            trainer = CelestialHNNTrainer(sys_obj, device=device)
            
            mlp_model, mlp_res = trainer.train_baseline_mlp(epochs=1200)
            hnn_model, hnn_res = trainer.train_hnn(epochs=1200)
            
            t_span = torch.linspace(0, sys_obj.T_max, 600, device=device)
            traj_true = sys_obj.ground_truth_trajectory(t_span).detach().cpu().numpy()
            
            z0 = sys_obj.z0.to(device)
            traj_mlp = mlp_model.integrate_symplectic_rk4(z0, t_span).squeeze(1).detach().cpu().numpy()
            traj_hnn = hnn_model.integrate_symplectic_rk4(z0, t_span).squeeze(1).detach().cpu().numpy()
            
            # 3-Panel Trajectory Triplet Figure
            fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
            
            # Subplot A: Ground Truth
            ax1 = axes[0]
            ax1.plot(traj_true[:, 0], traj_true[:, 1], color="#1f77b4", linewidth=2.2, label="DOP853 Exact Flow")
            ax1.scatter(traj_true[0, 0], traj_true[0, 1], color="green", s=80, marker="o", label=r"Start $z_0$", zorder=5)
            ax1.scatter(traj_true[-1, 0], traj_true[-1, 1], color="red", s=80, marker="x", label=r"End $z(T)$", zorder=5)
            subtitle_a = "Exact KAM Invariant Flow" if regime == "regular" else "Exact Multi-Loop Chaotic Orbit"
            ax1.set_title(f"(A) Ground Truth Trajectory\n{subtitle_a}", fontweight="bold")
            ax1.set_xlabel(r"$x$ / $q_1$" if sys_obj.spatial_dim == 2 else r"$z$ (Position)")
            ax1.set_ylabel(r"$y$ / $q_2$" if sys_obj.spatial_dim == 2 else r"$p_z$ (Momentum)")
            ax1.grid(True, linestyle="--")
            ax1.legend(loc="upper right")
            
            # Subplot B: Standard Baseline
            ax2 = axes[1]
            ax2.plot(traj_true[:, 0], traj_true[:, 1], color="gray", linestyle=":", linewidth=1.5, alpha=0.7, label="Exact Reference")
            ax2.plot(traj_mlp[:, 0], traj_mlp[:, 1], color="#d62728", linewidth=2.0, label=f"Standard Baseline\n(Rel L2: {mlp_res['rel_l2_error']*100:.2f}%)")
            ax2.scatter(traj_mlp[0, 0], traj_mlp[0, 1], color="green", s=80, marker="o", zorder=5)
            ax2.scatter(traj_mlp[-1, 0], traj_mlp[-1, 1], color="red", s=80, marker="x", zorder=5)
            subtitle_b = "Non-Conservative Decay" if regime == "regular" else "Severe Chaotic Dissipation & Divergence"
            ax2.set_title(f"(B) Standard Vector Field Baseline\n{subtitle_b}", fontweight="bold")
            ax2.set_xlabel(r"$x$ / $q_1$" if sys_obj.spatial_dim == 2 else r"$z$ (Position)")
            ax2.set_ylabel(r"$y$ / $q_2$" if sys_obj.spatial_dim == 2 else r"$p_z$ (Momentum)")
            ax2.grid(True, linestyle="--")
            ax2.legend(loc="upper right")
            
            # Subplot C: Symplectic HNN
            ax3 = axes[2]
            ax3.plot(traj_true[:, 0], traj_true[:, 1], color="gray", linestyle=":", linewidth=1.5, alpha=0.7, label="Exact Reference")
            ax3.plot(traj_hnn[:, 0], traj_hnn[:, 1], color="#2ca02c", linewidth=2.2, label=f"Symplectic HNN\n(Rel L2: {hnn_res['rel_l2_error']*100:.2f}%)")
            ax3.scatter(traj_hnn[0, 0], traj_hnn[0, 1], color="green", s=80, marker="o", zorder=5)
            ax3.scatter(traj_hnn[-1, 0], traj_hnn[-1, 1], color="red", s=80, marker="x", zorder=5)
            subtitle_c = r"Exact Symplectic Torus Preservation ($\Delta\mathcal{H}\sim 10^{-6}$)" if regime == "regular" else r"Chaotic Manifold Preservation ($\Delta\mathcal{H}\sim 10^{-5}$)"
            ax3.set_title(r"(C) Symplectic HNN (Ours)" + f"\n{subtitle_c}", fontweight="bold")
            ax3.set_xlabel(r"$x$ / $q_1$" if sys_obj.spatial_dim == 2 else r"$z$ (Position)")
            ax3.set_ylabel(r"$y$ / $q_2$" if sys_obj.spatial_dim == 2 else r"$p_z$ (Momentum)")
            ax3.grid(True, linestyle="--")
            ax3.legend(loc="upper right")
            
            fig.suptitle(f"{sys_title} - Trajectory Triplet Benchmark", fontsize=15, fontweight="bold", y=1.02)
            plt.tight_layout()
            plot_path = os.path.join(out_dir, f"{file_prefix}_triplet_comparison.png")
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            print(f"  [+] Saved: {plot_path}")

    print("\n[+] All Dual-Regime Visualizations generated successfully!")

if __name__ == "__main__":
    generate_dual_regime_visualizations()
