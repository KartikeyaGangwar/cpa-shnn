import torch
import numpy as np
from typing import Tuple, Optional
from scipy.integrate import solve_ivp

class EllipticSitnikovFiveBodySystem:
    """
    Non-Autonomous Celestial Problem:
    Elliptic Sitnikov Five-Body Problem Under Radiation Pressure
    Reference:
      M. Shahbaz Ullah, M. Javed Idrisi, and Vinay Kumar,
      "Elliptic Sitnikov five-body problem under radiation pressure",
      New Astronomy, Vol. 80, 101398, 2020.
      
    Physics:
      - 4 primary stars of mass M/4 move on Keplerian ellipses with eccentricity e in the (x, y) plane.
      - Radial barycentric distance breathes periodically: r(t) = a*(1 - e^2) / (1 + e*cos(nu(t)))
      - 5th infinitesimal body oscillates along vertical z-axis.
      - Radiation pressure parameter: q_rad in (0, 1].
      - State in Extended Phase Space: Z_ext = (z, t, p_z, p_t) in R^4.
    """
    def __init__(
        self,
        eccentricity: float = 0.25,
        q_radiation: float = 0.85,
        semi_major_axis: float = 1.0,
        regime: str = "chaotic",
        device: Optional[torch.device] = None
    ):
        self.name = "EllipticSitnikov_NonAutonomous"
        self.eccentricity = eccentricity
        self.q_rad = q_radiation
        self.a = semi_major_axis
        self.regime = regime
        self.spatial_dim = 1
        self.ext_spatial_dim = 2 # (z, t)
        self.state_dim = 4 # (z, t, p_z, p_t)
        self.device = device if device is not None else torch.device("cpu")
        
        # Orbital mean motion n = 1.0 (normalized units)
        self.n = 1.0
        self.p_orbit = 2.0 * np.pi / self.n
        self.T_max = 4.0 * self.p_orbit # 4 orbital periods
        
        if regime == "chaotic":
            self.z0_val = 1.45
            self.pz0_val = 0.35
        else:
            self.z0_val = 0.60
            self.pz0_val = 0.00
            
        # Compute exact initial Hamiltonian energy
        r0 = self.a * (1.0 - self.eccentricity) # Periastron at t=0
        V0 = -4.0 * self.q_rad / np.sqrt(self.z0_val ** 2 + r0 ** 2)
        H0 = 0.5 * (self.pz0_val ** 2) + V0
        self.pt0_val = -H0 # Arnold's conjugate energy constraint: K = H + p_t = 0
        
        self.z0 = torch.tensor([self.z0_val, 0.0, self.pz0_val, self.pt0_val], dtype=torch.float32, device=self.device)

    def _solve_kepler_true_anomaly(self, t_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solves Kepler equation M = E - e*sin(E) to find true anomaly nu(t) and r(t), dr/dt."""
        M = self.n * t_np
        E = M.copy()
        for _ in range(10):
            dE = (E - self.eccentricity * np.sin(E) - M) / (1.0 - self.eccentricity * np.cos(E))
            E -= dE
        
        # True anomaly nu
        cos_nu = (np.cos(E) - self.eccentricity) / (1.0 - self.eccentricity * np.cos(E))
        sin_nu = (np.sqrt(1.0 - self.eccentricity ** 2) * np.sin(E)) / (1.0 - self.eccentricity * np.cos(E))
        nu = np.arctan2(sin_nu, cos_nu)
        
        # Barycentric radius r(t)
        r = self.a * (1.0 - self.eccentricity ** 2) / (1.0 + self.eccentricity * cos_nu)
        
        # Radial velocity dr/dt
        dr_dt = (self.n * self.a * self.eccentricity * sin_nu) / np.sqrt(1.0 - self.eccentricity ** 2)
        return r, dr_dt, nu

    def exact_potential(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(-1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_np = t.detach().cpu().numpy().flatten()
        r_np, _, _ = self._solve_kepler_true_anomaly(t_np)
        r_t = torch.tensor(r_np, dtype=torch.float32, device=z.device).view_as(z)
        denom = torch.sqrt(z ** 2 + r_t ** 2)
        return -4.0 * self.q_rad / denom

    def exact_extended_hamiltonian(self, z_ext: torch.Tensor) -> torch.Tensor:
        z = z_ext[:, 0:1]
        t = z_ext[:, 1:2]
        pz = z_ext[:, 2:3]
        pt = z_ext[:, 3:4]
        V = self.exact_potential(z, t)
        kinetic = 0.5 * (pz ** 2)
        return kinetic + V + pt # Strictly 0.0 along exact trajectory

    def canonical_derivatives(self, z_ext: torch.Tensor) -> torch.Tensor:
        z = z_ext[:, 0:1]
        t = z_ext[:, 1:2]
        pz = z_ext[:, 2:3]
        pt = z_ext[:, 3:4]
        
        t_np = t.detach().cpu().numpy().flatten()
        r_np, dr_dt_np, _ = self._solve_kepler_true_anomaly(t_np)
        r_t = torch.tensor(r_np, dtype=torch.float32, device=z.device).view_as(z)
        dr_dt = torch.tensor(dr_dt_np, dtype=torch.float32, device=z.device).view_as(z)
        
        denom = (z ** 2 + r_t ** 2) ** 1.5
        # dz/dt = pz
        dz_dt = pz
        # dt/dt = 1.0 (Arnold Unit Clock)
        dt_dt = torch.ones_like(t)
        # dpz/dt = -∂V/∂z = -4*q_rad*z / denom
        dpz_dt = -4.0 * self.q_rad * z / denom
        # dpt/dt = -∂V/∂t = 4*q_rad*r(t)*dr_dt / denom (Power Exchange Rate)
        dpt_dt = 4.0 * self.q_rad * r_t * dr_dt / denom
        
        return torch.cat([dz_dt, dt_dt, dpz_dt, dpt_dt], dim=-1)

    def ground_truth_trajectory(self, t_eval: torch.Tensor) -> torch.Tensor:
        t_np = t_eval.detach().cpu().numpy()
        
        def ode_func(t, y):
            z_val, pz_val, pt_val = y
            r_np, dr_dt_np, _ = self._solve_kepler_true_anomaly(np.array([t]))
            r_t = r_np[0]
            dr_dt = dr_dt_np[0]
            denom = (z_val ** 2 + r_t ** 2) ** 1.5
            dz = pz_val
            dpz = -4.0 * self.q_rad * z_val / denom
            dpt = 4.0 * self.q_rad * r_t * dr_dt / denom
            return [dz, dpz, dpt]
            
        y0 = [self.z0_val, self.pz0_val, self.pt0_val]
        sol = solve_ivp(ode_func, [t_np[0], t_np[-1]], y0, t_eval=t_np, rtol=1e-11, atol=1e-13, method="DOP853")
        
        z_sol = sol.y[0]
        pz_sol = sol.y[1]
        pt_sol = sol.y[2]
        t_sol = t_np
        
        traj_np = np.column_stack([z_sol, t_sol, pz_sol, pt_sol])
        return torch.tensor(traj_np, dtype=torch.float32, device=self.device)

    def compute_trajectory_error(self, model) -> float:
        t_dense = torch.linspace(0, self.T_max, 2500, device=self.device)
        z_gt = self.ground_truth_trajectory(t_dense)
        z_pred = model.integrate_symplectic_rk4(self.z0, t_dense).squeeze(1)
        # Relative L2 error on spatial coordinate z
        num = torch.norm(z_pred[:, 0] - z_gt[:, 0])
        den = torch.norm(z_gt[:, 0]) + 1e-7
        return (num / den).item()
