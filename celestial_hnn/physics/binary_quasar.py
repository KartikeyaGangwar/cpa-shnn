import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List, Dict
from scipy.integrate import solve_ivp
from .base_hamiltonian import BaseHamiltonianSystem

class BinaryQuasarHamiltonianSystem(BaseHamiltonianSystem):
    """
    System I: Binary Quasar Canonical Hamiltonian Mechanics with Dual-Regime Support.
    - Regime 'regular': Quasi-periodic KAM torus orbit (T=3.0, clean sub-1% trajectory).
    - Regime 'chaotic': Long-horizon multi-loop saddle dynamics (T=8.0, chaotic invariant manifold).
    Reference: Kumar, V., Aggarwal, R., Sharma, P., Kaur, B. (New Astronomy, 2021: 101543)
    """
    def __init__(
        self,
        regime: str = "regular",
        mu: float = 0.30,
        n: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.regime = regime
        self.mu = mu
        self.n = n
        self.x1 = -mu
        self.x2 = 1.0 - mu
        
        if regime == "regular":
            self.eps1 = 0.25
            self.eps2 = 0.25
            self.T_max = 3.0
            self.x_init = (0.80, 0.30, 0.05, 0.45)
        else: # chaotic multi-loop
            self.eps1 = 0.10
            self.eps2 = 0.10
            self.T_max = 8.0
            self.x_init = (0.50, 0.20, 0.0, 0.40)
            
        self.eps1_sq = self.eps1 ** 2
        self.eps2_sq = self.eps2 ** 2
        
        x0, y0, vx0, vy0 = self.x_init
        px0 = vx0 - n * y0
        py0 = vy0 + n * x0
        z0_t = torch.tensor([x0, y0, px0, py0], dtype=torch.float32)
        
        super().__init__(
            name=f"BinaryQuasar_{regime}",
            spatial_dim=2,
            bounds_q=[(-2.0, 2.0), (-2.0, 2.0)],
            bounds_p=[(-2.5, 2.5), (-2.5, 2.5)],
            z0=z0_t,
            T_max=self.T_max,
            device=device,
        )
        self._precompute_reference_solution()

    def grav_potential(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r1_sq = (x - self.x1) ** 2 + y ** 2 + self.eps1_sq
        r2_sq = (x - self.x2) ** 2 + y ** 2 + self.eps2_sq
        return (1.0 - self.mu) / torch.sqrt(r1_sq) + self.mu / torch.sqrt(r2_sq)

    def grav_force(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r1_sq = (x - self.x1) ** 2 + y ** 2 + self.eps1_sq
        r2_sq = (x - self.x2) ** 2 + y ** 2 + self.eps2_sq
        r1_cube = r1_sq ** 1.5
        r2_cube = r2_sq ** 1.5
        fx = -(1.0 - self.mu) * (x - self.x1) / r1_cube - self.mu * (x - self.x2) / r2_cube
        fy = -(1.0 - self.mu) * y / r1_cube - self.mu * y / r2_cube
        return fx, fy

    def exact_hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        x = z[:, 0:1]
        y = z[:, 1:2]
        px = z[:, 2:3]
        py = z[:, 3:4]
        kinetic = 0.5 * (px ** 2 + py ** 2)
        coriolis_term = self.n * (px * y - py * x)
        phi = self.grav_potential(x, y)
        return kinetic + coriolis_term - phi

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
        
        r1_sq = (x - self.x1)**2 + y**2 + self.eps1_sq
        r2_sq = (x - self.x2)**2 + y**2 + self.eps2_sq
        fx = -(1.0 - self.mu) * (x - self.x1) / (r1_sq**1.5) - self.mu * (x - self.x2) / (r2_sq**1.5)
        fy = -(1.0 - self.mu) * y / (r1_sq**1.5) - self.mu * y / (r2_sq**1.5)
        
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
