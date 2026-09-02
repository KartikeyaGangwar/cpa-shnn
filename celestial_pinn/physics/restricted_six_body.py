import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List, Dict
from scipy.integrate import solve_ivp
from .base_celestial import BaseCelestialSystem

class RestrictedSixBodySquareSystem(BaseCelestialSystem):
    """
    System II: Restricted Six-Body Problem with Square Configuration
    Reference: Kumar, V., Idrisi, M. J., Ullah, M. S. (New Astronomy, 2021: 101451)
    """
    def __init__(
        self,
        m0: float = 0.20,
        a: float = 1.0,
        n: float = 1.0,
        eps: float = 0.10,
        T_max: float = 4.0,
        x_init: Tuple[float, float, float, float] = (0.35, 0.35, 0.25, -0.25),
        device: Optional[torch.device] = None,
    ):
        u0_tensor = torch.tensor(x_init, dtype=torch.float32)
        super().__init__(
            name="RestrictedSixBodySquareSystem",
            in_dim=1,
            out_dim=4,
            bounds=[(0.0, T_max)],
            u0=u0_tensor,
            device=device,
        )
        self.m0 = m0
        self.m = (1.0 - m0) / 4.0
        self.a = a
        self.n = n
        self.eps_sq = eps ** 2
        self.T_max = T_max
        self.x_init = x_init
        
        self.primaries = [
            (a, 0.0),
            (0.0, a),
            (-a, 0.0),
            (0.0, -a),
        ]
        
        # Exact initial Jacobi Constant
        x0, y0, vx0, vy0 = x_init
        r0_sq = x0**2 + y0**2 + self.eps_sq
        omega0 = 0.5 * (self.n**2) * (x0**2 + y0**2) + self.m0 / np.sqrt(r0_sq)
        for xi, yi in self.primaries:
            omega0 += self.m / np.sqrt((x0 - xi)**2 + (y0 - yi)**2 + self.eps_sq)
        self.C0 = 2.0 * omega0 - (vx0**2 + vy0**2)
        
        self._precompute_reference_solution()

    def potential(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        centrifugal = 0.5 * (self.n ** 2) * (x ** 2 + y ** 2)
        r0_sq = x ** 2 + y ** 2 + self.eps_sq
        grav_center = self.m0 / torch.sqrt(r0_sq)
        grav_corners = torch.zeros_like(x)
        for xi, yi in self.primaries:
            ri_sq = (x - xi) ** 2 + (y - yi) ** 2 + self.eps_sq
            grav_corners = grav_corners + self.m / torch.sqrt(ri_sq)
        return centrifugal + grav_center + grav_corners

    def potential_grad(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        omega_x = (self.n ** 2) * x
        omega_y = (self.n ** 2) * y
        r0_sq = x ** 2 + y ** 2 + self.eps_sq
        r0_cube = r0_sq ** 1.5
        omega_x = omega_x - self.m0 * x / r0_cube
        omega_y = omega_y - self.m0 * y / r0_cube
        for xi, yi in self.primaries:
            ri_sq = (x - xi) ** 2 + (y - yi) ** 2 + self.eps_sq
            ri_cube = ri_sq ** 1.5
            omega_x = omega_x - self.m * (x - xi) / ri_cube
            omega_y = omega_y - self.m * (y - yi) / ri_cube
        return omega_x, omega_y

    def jacobi_constant(self, x: torch.Tensor, y: torch.Tensor, vx: torch.Tensor, vy: torch.Tensor) -> torch.Tensor:
        v_sq = vx ** 2 + vy ** 2
        return 2.0 * self.potential(x, y) - v_sq

    def compute_energy_conservation_loss(self, model: nn.Module, t: torch.Tensor) -> torch.Tensor:
        u = model(t)
        x = u[:, 0:1]
        y = u[:, 1:2]
        vx = u[:, 2:3]
        vy = u[:, 3:4]
        c_t = self.jacobi_constant(x, y, vx, vy)
        return torch.mean((c_t - self.C0) ** 2)

    def sample_interior(self, n_samples: int) -> torch.Tensor:
        t = torch.linspace(0.0, self.T_max, n_samples, device=self.device).reshape(-1, 1)
        jitter = (torch.rand(n_samples, 1, device=self.device) - 0.5) * (self.T_max / n_samples)
        t_perturbed = torch.clamp(t + jitter, 0.0, self.T_max)
        t_perturbed.requires_grad_(True)
        return t_perturbed

    def compute_residuals(self, model: nn.Module, t: torch.Tensor) -> torch.Tensor:
        if not t.requires_grad:
            t = t.clone().detach().requires_grad_(True)
            
        u = model(t)
        x = u[:, 0:1]
        y = u[:, 1:2]
        vx = u[:, 2:3]
        vy = u[:, 3:4]
        
        grad_outputs = torch.ones_like(x)
        dx_dt = torch.autograd.grad(x, t, grad_outputs=grad_outputs, create_graph=True)[0]
        dy_dt = torch.autograd.grad(y, t, grad_outputs=grad_outputs, create_graph=True)[0]
        dvx_dt = torch.autograd.grad(vx, t, grad_outputs=grad_outputs, create_graph=True)[0]
        dvy_dt = torch.autograd.grad(vy, t, grad_outputs=grad_outputs, create_graph=True)[0]
        
        omega_x, omega_y = self.potential_grad(x, y)
        
        r1 = dx_dt - vx
        r2 = dy_dt - vy
        r3 = dvx_dt - (2.0 * self.n * vy + omega_x)
        r4 = dvy_dt - (-2.0 * self.n * vx + omega_y)
        
        return torch.cat([r1, r2, r3, r4], dim=-1)

    def _ode_rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        x, y, vx, vy = state
        omega_x = (self.n ** 2) * x
        omega_y = (self.n ** 2) * y
        r0_sq = x ** 2 + y ** 2 + self.eps_sq
        r0_cube = r0_sq ** 1.5
        omega_x -= self.m0 * x / r0_cube
        omega_y -= self.m0 * y / r0_cube
        for xi, yi in self.primaries:
            ri_sq = (x - xi) ** 2 + (y - yi) ** 2 + self.eps_sq
            ri_cube = ri_sq ** 1.5
            omega_x -= self.m * (x - xi) / ri_cube
            omega_y -= self.m * (y - yi) / ri_cube
        return [vx, vy, 2.0 * self.n * vy + omega_x, -2.0 * self.n * vx + omega_y]

    def _precompute_reference_solution(self):
        sol = solve_ivp(
            self._ode_rhs,
            (0.0, self.T_max),
            self.x_init,
            method="DOP853",
            rtol=1e-12,
            atol=1e-12,
            dense_output=True,
        )
        self.ref_interpolator = sol.sol

    def exact_solution(self, t: torch.Tensor) -> torch.Tensor:
        t_np = t.detach().cpu().numpy().ravel()
        u_np = self.ref_interpolator(t_np).T
        return torch.tensor(u_np, dtype=torch.float32, device=self.device)
