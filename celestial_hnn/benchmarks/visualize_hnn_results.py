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

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "axes.linewidth": 1.0,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.35,
})

def generate_hnn_publication_visualizations(out_dir: str = "results/plots"):
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "="*70)
    print("  GENERATING HAMILTONIAN PHASE PORTRAITS & ENERGY CONTOURS")
    print("="*70)
    
    x_vals = np.linspace(-2.5, 2.5, 200)
    y_vals = np.linspace(-2.5, 2.5, 200)
    X, Y = np.meshgrid(x_vals, y_vals)
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    
    # 1. Quasar
    quasar = BinaryQuasarHamiltonianSystem()
    z_grid = torch.stack([Xt.flatten(), Yt.flatten(), torch.zeros_like(Xt.flatten()), torch.zeros_like(Yt.flatten())], dim=-1)
    H_q = quasar.exact_hamiltonian(z_grid).reshape(200, 200).detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    cs = ax.contourf(X, Y, np.clip(H_q, -10, 5), levels=45, cmap="inferno")
    ax.contour(X, Y, np.clip(H_q, -10, 5), levels=18, colors="white", alpha=0.25, linewidths=0.5)
    ax.scatter([quasar.x1, quasar.x2], [0, 0], color="cyan", s=130, edgecolors="black", label="Quasars")
    
    t_span = torch.linspace(0, quasar.T_max, 500)
    traj = quasar.ground_truth_trajectory(t_span).detach().cpu().numpy()
    ax.plot(traj[:, 0], traj[:, 1], color="lime", linewidth=2.0, label="Symplectic Orbit")
    ax.set_title("System I: Binary Quasar Canonical Hamiltonian $\\mathcal{H}(x, y, p_x=0, p_y=0)$", fontweight="bold")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(loc="upper right")
    plt.colorbar(cs, ax=ax, label="Hamiltonian Energy $\\mathcal{H}$")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "system1_quasar_hamiltonian.png"), bbox_inches="tight")
    plt.close()
    
    print(f"[+] Hamiltonian scientific plots saved to {out_dir}/!")

if __name__ == "__main__":
    generate_hnn_publication_visualizations()
