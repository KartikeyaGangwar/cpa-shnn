import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List, Dict
from scipy.integrate import solve_ivp
from .base_hamiltonian import BaseHamiltonianSystem

class RestrictedSixBodyHamiltonianSystem(BaseHamiltonianSystem):
    """
    System II: Canonical Restricted Six-Body Problem with Square Configuration and Dual Regime.
    Reference: Aggarwal, R., Mittal, A., Kumar, V., & Suraj, M. S. (Astrophys Space Sci, 2018: 363:104)
    """
    def __init__(
        self,
        regime: str = "regular",
        mass_central: float = 0.50,
        mass_primaries: float = 0.125,
        n: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.regime = regime
        self.m0 = mass_central
        self.m_p = mass_primaries
        self.n = n
        
        if regime == "regular":
            self.eps = 0.25
            self.T_max = 3.0
            self.x_init = (0.35, 0.0, 0.0, 0.45)
        else: # chaotic multi-loop
            self.eps = 0.10
            self.T_max = 8.0
            self.x_init = (0.50, 0.50, 0.0, 0.50)
            
        self.eps_sq = self.eps ** 2
        
        a = 1.0 / np.sqrt(2.0)
        self.primaries_pos = [
            (a, a),
            (-a, a),
            (-a, -a),
            (a, -a),
        ]
        
        x0, y0, vx0, vy0 = self.x_init
        px0 = vx0 - n * y0
        py0 = vy0 + n * x0
        z0_t = torch.tensor([x0, y0, px0, py0], dtype=torch.float32)
        
        super().__init__(
            name=f"RestrictedSixBody_{regime}",
            spatial_dim=2,
            bounds_q=[(-2.0, 2.0), (-2.0, 2.0)],
            bounds_p=[(-2.5, 2.5), (-2.5, 2.5)],
            z0=z0_t,
            T_max=self.T_max,
            device=device,
        )
        self._precompute_reference_solution()

    def grav_potential(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r0_sq = x ** 2 + y ** 2 + self.eps_sq
        phi = self.m0 / torch.sqrt(r0_sq)
        for xp, yp in self.primaries_pos:
            rp_sq = (x - xp) ** 2 + (y - yp) ** 2 + self.eps_sq
            phi = phi + self.m_p / torch.sqrt(rp_sq)
        return phi

    def grav_force(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r0_sq = x ** 2 + y ** 2 + self.eps_sq
        r0_cube = r0_sq ** 1.5
        fx = -self.m0 * x / r0_cube
        fy = -self.m0 * y / r0_cube
        for xp, yp in self.primaries_pos:
            rp_sq = (x - xp) ** 2 + (y - yp) ** 2 + self.eps_sq
            rp_cube = rp_sq ** 1.5
            fx = fx - self.m_p * (x - xp) / rp_cube
            fy = fy - self.m_p * (y - yp) / rp_cube
        return fx, fy

    def exact_hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        x = z[:, 0:1]
        y = z[:, 1:2]
        px = z[:, 2:3]
        py = z[:, 3:4]
        kinetic = 0.5 * (px ** 2 + py ** 2)
        coriolis = self.n * (px * y - py * x)
        phi = self.grav_potential(x, y)
        return kinetic + coriolis - phi

    def canonical_derivatives(self, z: torch.Tensor) -> torch.Tensor:
        x = z[:, 0:1]
        y = z[:, 1:2]
        px = z[:, 2:3]
        py = z[:, 3:4]
        
        dx_dt = px + self.n * y
        dy_dt = py - self.n * x
        
        fx, fy = self.grav_force(x, y)
        dpx_dt = self.n * py + fx
        dpy_dt = -self.n * px + fy
        return torch.cat([dx_dt, dy_dt, dpx_dt, dpy_dt], dim=-1)

    def _ode_rhs(self, t: float, z_np: np.ndarray) -> np.ndarray:
        x, y, px, py = z_np
        dx_dt = px + self.n * y
        dy_dt = py - self.n * x
        
        r0_sq = x**2 + y**2 + self.eps_sq
        r0_cube = r0_sq ** 1.5
        fx = -self.m0 * x / r0_cube
        fy = -self.m0 * y / r0_cube
        for xp, yp in self.primaries_pos:
            rp_sq = (x - xp)**2 + (y - yp)**2 + self.eps_sq
            rp_cube = rp_sq ** 1.5
            fx -= self.m_p * (x - xp) / rp_cube
            fy -= self.m_p * (y - yp) / rp_cube
            
        dpx_dt = self.n * py + fx
        dpy_dt = -self.n * px + fy
        return [dx_dt, dy_dt, dpx_dt, dpy_dt]

    def _precompute_reference_solution(self):
        x0, y0, vx0, vy0 = self.x_init
        px0 = vx0 - self.n * y0
        py0 = vy0 + self.n * x0
        sol = solve_ivp(
            self._ode_rhs,
            (0.0, self.T_max),
            [x0, y0, px0, py0],
            method="DOP853",
            rtol=1e-12,
            atol=1e-12,
            dense_output=True,
        )
        self.ref_interpolator = sol.sol

    def ground_truth_trajectory(self, t_span: torch.Tensor) -> torch.Tensor:
        t_np = t_span.detach().cpu().numpy().ravel()
        z_np = self.ref_interpolator(t_np).T
        return torch.tensor(z_np, dtype=torch.float32, device=self.device)
