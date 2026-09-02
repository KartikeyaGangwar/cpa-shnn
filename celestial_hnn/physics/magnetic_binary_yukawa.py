import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List, Dict
from scipy.integrate import solve_ivp
from .base_hamiltonian import BaseHamiltonianSystem

class MagneticYukawaHamiltonianSystem(BaseHamiltonianSystem):
    """
    System IV: Photogravitational Magnetic Binary with Non-Newtonian Yukawa Fifth-Force
    Reference: Kumar, V., Aggarwal, R., Marig, S. K. (Astronomy and Computing, 2023: 100783)
    """
    def __init__(
        self,
        mu: float = 0.35,
        q1: float = 0.90,
        q2: float = 0.85,
        alpha: float = 0.50,
        lambda_y: float = 0.40,
        M1: float = 0.05,
        M2: float = 0.03,
        n: float = 1.0,
        eps: float = 0.15,
        T_max: float = 6.0,
        x_init: Tuple[float, float, float, float] = (0.45, 0.30, 0.10, 0.35),
        device: Optional[torch.device] = None,
    ):
        self.mu = mu
        self.mu1 = 1.0 - mu
        self.mu2 = mu
        self.q1 = q1
        self.q2 = q2
        self.alpha = alpha
        self.lambda_y = lambda_y
        self.M1 = M1
        self.M2 = M2
        self.n = n
        self.eps = eps
        self.eps_sq = eps ** 2
        self.x1 = -mu
        self.x2 = 1.0 - mu
        self.x_init = x_init
        
        x0, y0, vx0, vy0 = x_init
        px0 = vx0 - n * y0
        py0 = vy0 + n * x0
        z0_t = torch.tensor([x0, y0, px0, py0], dtype=torch.float32)
        
        super().__init__(
            name="MagneticYukawaHamiltonianSystem",
            spatial_dim=2,
            bounds_q=[(-2.0, 2.0), (-2.0, 2.0)],
            bounds_p=[(-2.0, 2.0), (-2.0, 2.0)],
            z0=z0_t,
            T_max=T_max,
            device=device,
        )
        self._precompute_reference_solution()

    def sample_phase_space(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples phase space ensuring points stay outside the hard collision core (r > eps)."""
        valid_z = []
        while len(valid_z) < n_samples:
            batch_size = n_samples * 2
            x = (self.bounds_q[0][1] - self.bounds_q[0][0]) * torch.rand(batch_size, 1, device=self.device) + self.bounds_q[0][0]
            y = (self.bounds_q[1][1] - self.bounds_q[1][0]) * torch.rand(batch_size, 1, device=self.device) + self.bounds_q[1][0]
            px = (self.bounds_p[0][1] - self.bounds_p[0][0]) * torch.rand(batch_size, 1, device=self.device) + self.bounds_p[0][0]
            py = (self.bounds_p[1][1] - self.bounds_p[1][0]) * torch.rand(batch_size, 1, device=self.device) + self.bounds_p[1][0]
            
            r1 = torch.sqrt((x - self.x1)**2 + y**2)
            r2 = torch.sqrt((x - self.x2)**2 + y**2)
            mask = (r1 > self.eps) & (r2 > self.eps)
            
            z_batch = torch.cat([x[mask].reshape(-1, 1), y[mask].reshape(-1, 1), px[mask].reshape(-1, 1), py[mask].reshape(-1, 1)], dim=-1)
            valid_z.append(z_batch)
            
        z = torch.cat(valid_z, dim=0)[:n_samples]
        dz_dt = self.canonical_derivatives(z)
        return z, dz_dt

    def coupled_potential(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r1_sq = (x - self.x1) ** 2 + y ** 2 + self.eps_sq
        r2_sq = (x - self.x2) ** 2 + y ** 2 + self.eps_sq
        r1 = torch.sqrt(r1_sq)
        r2 = torch.sqrt(r2_sq)
        
        yukawa1 = (self.q1 * self.mu1 / r1) * (1.0 + self.alpha * torch.exp(-r1 / self.lambda_y))
        yukawa2 = (self.q2 * self.mu2 / r2) * (1.0 + self.alpha * torch.exp(-r2 / self.lambda_y))
        mag1 = self.M1 / (r1_sq * r1)
        mag2 = self.M2 / (r2_sq * r2)
        return yukawa1 + yukawa2 + mag1 + mag2

    def coupled_force(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r1_sq = (x - self.x1) ** 2 + y ** 2 + self.eps_sq
        r2_sq = (x - self.x2) ** 2 + y ** 2 + self.eps_sq
        r1 = torch.sqrt(r1_sq)
        r2 = torch.sqrt(r2_sq)
        
        exp1 = torch.exp(-r1 / self.lambda_y)
        d_tot1_dr = -(self.q1 * self.mu1 / r1_sq) * (1.0 + self.alpha * (1.0 + r1 / self.lambda_y) * exp1) - 3.0 * self.M1 / (r1_sq ** 2)
        exp2 = torch.exp(-r2 / self.lambda_y)
        d_tot2_dr = -(self.q2 * self.mu2 / r2_sq) * (1.0 + self.alpha * (1.0 + r2 / self.lambda_y) * exp2) - 3.0 * self.M2 / (r2_sq ** 2)
        
        fx = d_tot1_dr * (x - self.x1) / r1 + d_tot2_dr * (x - self.x2) / r2
        fy = d_tot1_dr * y / r1 + d_tot2_dr * y / r2
        return fx, fy

    def exact_hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        x = z[:, 0:1]
        y = z[:, 1:2]
        px = z[:, 2:3]
        py = z[:, 3:4]
        kinetic = 0.5 * (px ** 2 + py ** 2)
        coriolis = self.n * (px * y - py * x)
        phi = self.coupled_potential(x, y)
        return kinetic + coriolis - phi

    def canonical_derivatives(self, z: torch.Tensor) -> torch.Tensor:
        x = z[:, 0:1]
        y = z[:, 1:2]
        px = z[:, 2:3]
        py = z[:, 3:4]
        
        dx_dt = px + self.n * y
        dy_dt = py - self.n * x
        
        fx, fy = self.coupled_force(x, y)
        dpx_dt = self.n * py + fx
        dpy_dt = -self.n * px + fy
        return torch.cat([dx_dt, dy_dt, dpx_dt, dpy_dt], dim=-1)

    def _ode_rhs(self, t: float, z_np: np.ndarray) -> np.ndarray:
        x, y, px, py = z_np
        dx_dt = px + self.n * y
        dy_dt = py - self.n * x
        
        r1_sq = (x - self.x1)**2 + y**2 + self.eps_sq
        r2_sq = (x - self.x2)**2 + y**2 + self.eps_sq
        r1 = np.sqrt(r1_sq)
        r2 = np.sqrt(r2_sq)
        exp1 = np.exp(-r1 / self.lambda_y)
        d_tot1_dr = -(self.q1 * self.mu1 / r1_sq) * (1.0 + self.alpha * (1.0 + r1 / self.lambda_y) * exp1) - 3.0 * self.M1 / (r1_sq ** 2)
        exp2 = np.exp(-r2 / self.lambda_y)
        d_tot2_dr = -(self.q2 * self.mu2 / r2_sq) * (1.0 + self.alpha * (1.0 + r2 / self.lambda_y) * exp2) - 3.0 * self.M2 / (r2_sq ** 2)
        fx = d_tot1_dr * (x - self.x1) / r1 + d_tot2_dr * (x - self.x2) / r2
        fy = d_tot1_dr * y / r1 + d_tot2_dr * y / r2
        
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
