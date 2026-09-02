import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List, Dict
from scipy.integrate import solve_ivp
from .base_hamiltonian import BaseHamiltonianSystem

class RestrictedSixBodyHamiltonianSystem(BaseHamiltonianSystem):
    """
    System II: Restricted Six-Body Problem in Canonical Hamiltonian Mechanics
    Reference: Kumar, V., Idrisi, M. J., Ullah, M. S. (New Astronomy, 2021: 101451)
    """
    def __init__(
        self,
        m0: float = 0.20,
        a: float = 1.0,
        n: float = 1.0,
        eps: float = 0.10,
        T_max: float = 6.0,
        x_init: Tuple[float, float, float, float] = (0.35, 0.35, 0.25, -0.25),
        device: Optional[torch.device] = None,
    ):
        self.m0 = m0
        self.m = (1.0 - m0) / 4.0
        self.a = a
        self.n = n
        self.eps_sq = eps ** 2
        self.x_init = x_init
        self.primaries = [
            (a, 0.0),
            (0.0, a),
            (-a, 0.0),
            (0.0, -a),
        ]
        
        x0, y0, vx0, vy0 = x_init
        px0 = vx0 - n * y0
        py0 = vy0 + n * x0
        z0_t = torch.tensor([x0, y0, px0, py0], dtype=torch.float32)
        
        super().__init__(
            name="RestrictedSixBodyHamiltonianSystem",
            spatial_dim=2,
            bounds_q=[(-2.0, 2.0), (-2.0, 2.0)],
            bounds_p=[(-2.0, 2.0), (-2.0, 2.0)],
            z0=z0_t,
            T_max=T_max,
            device=device,
        )
        self._precompute_reference_solution()

    def grav_potential(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r0_sq = x ** 2 + y ** 2 + self.eps_sq
        phi = self.m0 / torch.sqrt(r0_sq)
        for xi, yi in self.primaries:
            ri_sq = (x - xi) ** 2 + (y - yi) ** 2 + self.eps_sq
            phi = phi + self.m / torch.sqrt(ri_sq)
        return phi

    def grav_force(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r0_sq = x ** 2 + y ** 2 + self.eps_sq
        r0_cube = r0_sq ** 1.5
        fx = -self.m0 * x / r0_cube
        fy = -self.m0 * y / r0_cube
        for xi, yi in self.primaries:
            ri_sq = (x - xi) ** 2 + (y - yi) ** 2 + self.eps_sq
            ri_cube = ri_sq ** 1.5
            fx = fx - self.m * (x - xi) / ri_cube
            fy = fy - self.m * (y - yi) / ri_cube
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
        fx = -self.m0 * x / (r0_sq**1.5)
        fy = -self.m0 * y / (r0_sq**1.5)
        for xi, yi in self.primaries:
            ri_sq = (x - xi)**2 + (y - yi)**2 + self.eps_sq
            fx -= self.m * (x - xi) / (ri_sq**1.5)
            fy -= self.m * (y - yi) / (ri_sq**1.5)
            
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
