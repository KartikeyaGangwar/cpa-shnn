import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List, Dict
from scipy.integrate import solve_ivp
from .base_celestial import BaseCelestialSystem

class PhotogravitationalMagneticYukawaBinary(BaseCelestialSystem):
    """
    System IV: Photogravitational Magnetic Binary with Yukawa Fifth-Force Correction
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
        eps: float = 0.10,
        T_max: float = 4.0,
        x_init: Tuple[float, float, float, float] = (0.45, 0.30, 0.10, 0.35),
        device: Optional[torch.device] = None,
    ):
        u0_tensor = torch.tensor(x_init, dtype=torch.float32)
        super().__init__(
            name="PhotogravitationalMagneticYukawaBinary",
            in_dim=1,
            out_dim=4,
            bounds=[(0.0, T_max)],
            u0=u0_tensor,
            device=device,
        )
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
        self.eps_sq = eps ** 2
        self.x1 = -mu
        self.x2 = 1.0 - mu
        self.T_max = T_max
        self.x_init = x_init
        
        # Exact initial energy
        x0, y0, vx0, vy0 = x_init
        r1_0 = np.sqrt((x0 - self.x1)**2 + y0**2 + self.eps_sq)
        r2_0 = np.sqrt((x0 - self.x2)**2 + y0**2 + self.eps_sq)
        cent0 = 0.5 * (self.n**2) * (x0**2 + y0**2)
        yuk1 = (self.q1 * self.mu1 / r1_0) * (1.0 + self.alpha * np.exp(-r1_0 / self.lambda_y))
        yuk2 = (self.q2 * self.mu2 / r2_0) * (1.0 + self.alpha * np.exp(-r2_0 / self.lambda_y))
        mag1 = self.M1 / (r1_0 ** 3)
        mag2 = self.M2 / (r2_0 ** 3)
        omega0 = cent0 + yuk1 + yuk2 + mag1 + mag2
        self.C0 = 2.0 * omega0 - (vx0**2 + vy0**2)
        
        self._precompute_reference_solution()

    def potential(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r1_sq = (x - self.x1) ** 2 + y ** 2 + self.eps_sq
        r2_sq = (x - self.x2) ** 2 + y ** 2 + self.eps_sq
        r1 = torch.sqrt(r1_sq)
        r2 = torch.sqrt(r2_sq)
        centrifugal = 0.5 * (self.n ** 2) * (x ** 2 + y ** 2)
        yukawa1 = (self.q1 * self.mu1 / r1) * (1.0 + self.alpha * torch.exp(-r1 / self.lambda_y))
        yukawa2 = (self.q2 * self.mu2 / r2) * (1.0 + self.alpha * torch.exp(-r2 / self.lambda_y))
        mag1 = self.M1 / (r1_sq * r1)
        mag2 = self.M2 / (r2_sq * r2)
        return centrifugal + yukawa1 + yukawa2 + mag1 + mag2

    def potential_grad(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r1_sq = (x - self.x1) ** 2 + y ** 2 + self.eps_sq
        r2_sq = (x - self.x2) ** 2 + y ** 2 + self.eps_sq
        r1 = torch.sqrt(r1_sq)
        r2 = torch.sqrt(r2_sq)
        omega_x = (self.n ** 2) * x
        omega_y = (self.n ** 2) * y
        
        exp1 = torch.exp(-r1 / self.lambda_y)
        d_tot1_dr = -(self.q1 * self.mu1 / r1_sq) * (1.0 + self.alpha * (1.0 + r1 / self.lambda_y) * exp1) - 3.0 * self.M1 / (r1_sq ** 2)
        exp2 = torch.exp(-r2 / self.lambda_y)
        d_tot2_dr = -(self.q2 * self.mu2 / r2_sq) * (1.0 + self.alpha * (1.0 + r2 / self.lambda_y) * exp2) - 3.0 * self.M2 / (r2_sq ** 2)
        
        omega_x = omega_x + d_tot1_dr * (x - self.x1) / r1 + d_tot2_dr * (x - self.x2) / r2
        omega_y = omega_y + d_tot1_dr * y / r1 + d_tot2_dr * y / r2
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
        r1_sq = (x - self.x1)**2 + y**2 + self.eps_sq
        r2_sq = (x - self.x2)**2 + y**2 + self.eps_sq
        r1 = np.sqrt(r1_sq)
        r2 = np.sqrt(r2_sq)
        omega_x = (self.n ** 2) * x
        omega_y = (self.n ** 2) * y
        exp1 = np.exp(-r1 / self.lambda_y)
        d_tot1_dr = -(self.q1 * self.mu1 / r1_sq) * (1.0 + self.alpha * (1.0 + r1 / self.lambda_y) * exp1) - 3.0 * self.M1 / (r1_sq ** 2)
        exp2 = np.exp(-r2 / self.lambda_y)
        d_tot2_dr = -(self.q2 * self.mu2 / r2_sq) * (1.0 + self.alpha * (1.0 + r2 / self.lambda_y) * exp2) - 3.0 * self.M2 / (r2_sq ** 2)
        omega_x += d_tot1_dr * (x - self.x1) / r1 + d_tot2_dr * (x - self.x2) / r2
        omega_y += d_tot1_dr * y / r1 + d_tot2_dr * y / r2
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
