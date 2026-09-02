import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from celestial_pinn.physics.binary_quasar import BinaryQuasarSystem
from celestial_pinn.physics.restricted_six_body import RestrictedSixBodySquareSystem
from celestial_pinn.physics.sitnikov_five_body import EllipticSitnikovFiveBodySystem
from celestial_pinn.physics.magnetic_binary_yukawa import PhotogravitationalMagneticYukawaBinary
from celestial_pinn.solvers.basin_analyzer import BasinEntropyAnalyzer

# Configure LaTeX-style typography and clean aesthetics
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

def generate_publication_visualizations(out_dir: str = "results/plots"):
    os.makedirs(out_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("  GENERATING HIGH-RESOLUTION CELESTIAL SCIENTIFIC VISUALIZATIONS (LaTeX CM Style)")
    print("="*70)
    
    x_vals = np.linspace(-2.5, 2.5, 300)
    y_vals = np.linspace(-2.5, 2.5, 300)
    X, Y = np.meshgrid(x_vals, y_vals)
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    
    # 1. Quasar Phase Space & Effective Potential Contours
    print("[1/4] Generating Binary Quasar Phase Portrait & Potential...")
    quasar = BinaryQuasarSystem()
    Omega = quasar.potential(Xt, Yt).detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(8, 7.2), dpi=300)
    cs = ax.contourf(X, Y, np.clip(Omega, -5, 15), levels=45, cmap="inferno")
    ax.contour(X, Y, np.clip(Omega, -5, 15), levels=18, colors="white", alpha=0.25, linewidths=0.5)
    
    p1 = ax.scatter([quasar.x1, quasar.x2], [0, 0], color="cyan", s=130, edgecolors="black", zorder=5, label="Quasar Primaries")
    
    t_span = torch.linspace(0, quasar.T_max, 500).reshape(-1, 1)
    traj = quasar.exact_solution(t_span).detach().cpu().numpy()
    p2, = ax.plot(traj[:, 0], traj[:, 1], color="lime", linewidth=2.0, label="DOP853 Exact Trajectory", zorder=4)
    p3 = ax.scatter([traj[0, 0]], [traj[0, 1]], color="yellow", s=90, edgecolors="black", zorder=6, label="Initial State")
    
    ax.set_title("System I: Binary Quasar Potential Field $\\Omega_p(x, y)$ & Chaotic Orbit", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("$x$ (dimensionless)")
    ax.set_ylabel("$y$ (dimensionless)")
    ax.grid(True, linestyle="--", alpha=0.3)
    
    cbar = plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Effective Potential $\\Omega_p(x, y)$", rotation=270, labelpad=15)
    
    ax.legend(handles=[p1, p2, p3], loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=True, fancybox=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "system1_quasar_field_trajectory.png"), bbox_inches="tight")
    plt.close()
    
    # 2. Restricted 6-Body Square Geometry & Multi-Primary Wells
    print("[2/4] Generating Restricted 6-Body Square Potential...")
    sixbody = RestrictedSixBodySquareSystem()
    Omega_6 = sixbody.potential(Xt, Yt).detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(8, 7.2), dpi=300)
    cs2 = ax.contourf(X, Y, np.clip(Omega_6, -5, 20), levels=50, cmap="magma")
    ax.contour(X, Y, np.clip(Omega_6, -5, 20), levels=20, colors="white", alpha=0.25, linewidths=0.5)
    
    prim_x = [p[0] for p in sixbody.primaries] + [0]
    prim_y = [p[1] for p in sixbody.primaries] + [0]
    p1 = ax.scatter(prim_x, prim_y, color="deepskyblue", s=130, edgecolors="white", linewidths=1.2, zorder=5, label="5 Primaries (Square + Central)")
    
    traj6 = sixbody.exact_solution(torch.linspace(0, sixbody.T_max, 500).reshape(-1, 1)).detach().cpu().numpy()
    p2, = ax.plot(traj6[:, 0], traj6[:, 1], color="yellow", linewidth=2.0, label="6th-Body Chaotic Trajectory", zorder=4)
    p3 = ax.scatter([traj6[0, 0]], [traj6[0, 1]], color="red", s=90, edgecolors="black", zorder=6, label="Initial Injection")
    
    ax.set_title("System II: Restricted Six-Body Problem with Square Configuration", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("$x$ (dimensionless)")
    ax.set_ylabel("$y$ (dimensionless)")
    ax.grid(True, linestyle="--", alpha=0.3)
    
    cbar2 = plt.colorbar(cs2, ax=ax, fraction=0.046, pad=0.04)
    cbar2.set_label("Effective Potential $\\Omega_p(x, y)$", rotation=270, labelpad=15)
    
    ax.legend(handles=[p1, p2, p3], loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=True, fancybox=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "system2_sixbody_geometry.png"), bbox_inches="tight")
    plt.close()
    
    # 3. Sitnikov Five-Body Non-Autonomous True Anomaly Dynamics
    print("[3/4] Generating Sitnikov 5-Body Oscillatory Trajectory...")
    sitnikov = EllipticSitnikovFiveBodySystem(eccentricity=0.25, radiation_q=0.80)
    v_dense = torch.linspace(0, sitnikov.V_max, 600).reshape(-1, 1)
    sol_sit = sitnikov.exact_solution(v_dense).detach().cpu().numpy()
    r_prim = sitnikov.orbital_radius(v_dense).detach().cpu().numpy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.5), dpi=300, sharex=True)
    
    l1, = ax1.plot(v_dense.numpy(), sol_sit[:, 0], color="#d62728", linewidth=2.2, label="Vertical Position $z(v)$")
    l2, = ax1.plot(v_dense.numpy(), sol_sit[:, 1], color="#1f77b4", linestyle="--", linewidth=1.8, label="Velocity $v_z(v) = \\frac{dz}{dv}$")
    ax1.set_ylabel("Oscillatory State")
    ax1.set_title("System III: Elliptic Sitnikov Five-Body Dynamics ($e=0.25, q=0.80$)", fontsize=12, fontweight="bold", pad=10)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(handles=[l1, l2], loc="upper right", frameon=True)
    
    l3, = ax2.plot(v_dense.numpy(), r_prim, color="#ff7f0e", linewidth=2.0, label="Instantaneous Primary Orbit Radius $r(v) = \\frac{1-e^2}{1+e\\cos(v)}$")
    ax2.set_xlabel("True Anomaly $v$ (radians)")
    ax2.set_ylabel("Pulsating Radius $r(v)$")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(handles=[l3], loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "system3_sitnikov_trajectory.png"), bbox_inches="tight")
    plt.close()
    
    # 4. Multi-Physics Magnetic Yukawa Binary System
    print("[4/4] Generating Magnetic Yukawa Binary Multi-Scale Field...")
    yukawa = PhotogravitationalMagneticYukawaBinary(alpha=0.60, lambda_y=0.40)
    Omega_y = yukawa.potential(Xt, Yt).detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(8, 7.2), dpi=300)
    cs4 = ax.contourf(X, Y, np.clip(Omega_y, -5, 25), levels=50, cmap="viridis")
    ax.contour(X, Y, np.clip(Omega_y, -5, 25), levels=20, colors="white", alpha=0.25, linewidths=0.5)
    
    p1 = ax.scatter([yukawa.x1, yukawa.x2], [0, 0], color="red", s=140, edgecolors="white", linewidths=1.2, zorder=5, label="Magnetized Yukawa Primaries")
    
    circle1 = plt.Circle((yukawa.x1, 0), yukawa.lambda_y, color="yellow", fill=False, linestyle="--", linewidth=2.0, label="Yukawa Horizon ($\\lambda=0.40$)")
    circle2 = plt.Circle((yukawa.x2, 0), yukawa.lambda_y, color="yellow", fill=False, linestyle="--", linewidth=2.0)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    
    trajy = yukawa.exact_solution(torch.linspace(0, yukawa.T_max, 500).reshape(-1, 1)).detach().cpu().numpy()
    p2, = ax.plot(trajy[:, 0], trajy[:, 1], color="cyan", linewidth=2.0, label="Coupled 4-Force Orbit", zorder=4)
    
    ax.set_title("System IV: Photogravitational Magnetic Binary with Yukawa Fifth-Force", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("$x$ (dimensionless)")
    ax.set_ylabel("$y$ (dimensionless)")
    ax.grid(True, linestyle="--", alpha=0.3)
    
    cbar4 = plt.colorbar(cs4, ax=ax, fraction=0.046, pad=0.04)
    cbar4.set_label("Effective Potential $\\Omega_p(x, y)$", rotation=270, labelpad=15)
    
    ax.legend(handles=[p1, circle1, p2], loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=True, fancybox=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "system4_magnetic_yukawa_field.png"), bbox_inches="tight")
    plt.close()
    
    print(f"\n[+] All 4 LaTeX-styled scientific visualizations saved to {out_dir}/!")

if __name__ == "__main__":
    generate_publication_visualizations()
