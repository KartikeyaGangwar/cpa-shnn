import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List, Dict
from scipy.integrate import solve_ivp
from .base_celestial import BaseCelestialSystem

class EllipticSitnikovFiveBodySystem(BaseCelestialSystem):
    """
    System III: Elliptic Sitnikov Five-Body Problem Under Radiation Pressure
    Reference: Ullah, M. S., Idrisi, M. J., Kumar, V. (New Astronomy, 2020: 101398)
    """
    def __init__(
        self,
        eccentricity: float = 0.20,
        radiation_q: float = 0.85,
        V_max: float = 6.283185, # ~1 full orbit (2*pi)
        z_init: Tuple[float, float] = (0.50, 0.0),
        device: Optional[torch.device] = None,
    ):
        u0_tensor = torch.tensor(z_init, dtype=torch.float32)
        super().__init__(
            name="EllipticSitnikovFiveBodySystem",
            in_dim=1,
            out_dim=2,
            bounds=[(0.0, V_max)],
            u0=u0_tensor,
            device=device,
        )
        self.e = eccentricity
        self.q = radiation_q
        self.V_max = V_max
        self.z_init = z_init
        self._precompute_reference_solution()

    def orbital_radius(self, v: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.e ** 2) / (1.0 + self.e * torch.cos(v))

    def sample_interior(self, n_samples: int) -> torch.Tensor:
        v = torch.linspace(0.0, self.V_max, n_samples, device=self.device).reshape(-1, 1)
        jitter = (torch.rand(n_samples, 1, device=self.device) - 0.5) * (self.V_max / n_samples)
        v_perturbed = torch.clamp(v + jitter, 0.0, self.V_max)
        v_perturbed.requires_grad_(True)
        return v_perturbed

    def compute_residuals(self, model: nn.Module, v: torch.Tensor) -> torch.Tensor:
        if not v.requires_grad:
            v = v.clone().detach().requires_grad_(True)
            
        u = model(v)
        z = u[:, 0:1]
        vz = u[:, 1:2]
        
        grad_outputs = torch.ones_like(z)
        dz_dv = torch.autograd.grad(z, v, grad_outputs=grad_outputs, create_graph=True)[0]
        dvz_dv = torch.autograd.grad(vz, v, grad_outputs=grad_outputs, create_graph=True)[0]
        
        cos_v = torch.cos(v)
        sin_v = torch.sin(v)
        denom_prim = 1.0 + self.e * cos_v
        r_v = (1.0 - self.e ** 2) / denom_prim
        
        denom_force = (z ** 2 + 0.5 * (r_v ** 2)) ** 1.5
        grav_rad_force = 4.0 * self.q * z / denom_force
        
        r1 = dz_dv - vz
        r2 = denom_prim * dvz_dv - 2.0 * self.e * sin_v * vz + grav_rad_force
        
        return torch.cat([r1, r2], dim=-1)

    def _ode_rhs(self, v: float, state: np.ndarray) -> np.ndarray:
        z, vz = state
        cos_v = np.cos(v)
        sin_v = np.sin(v)
        denom_prim = 1.0 + self.e * cos_v
        r_v = (1.0 - self.e ** 2) / denom_prim
        denom_force = (z ** 2 + 0.5 * (r_v ** 2)) ** 1.5
        grav_rad_force = 4.0 * self.q * z / denom_force
        return [vz, (2.0 * self.e * sin_v * vz - grav_rad_force) / denom_prim]

    def _precompute_reference_solution(self):
        sol = solve_ivp(
            self._ode_rhs,
            (0.0, self.V_max),
            self.z_init,
            method="DOP853",
            rtol=1e-12,
            atol=1e-12,
            dense_output=True,
        )
        self.ref_interpolator = sol.sol

    def exact_solution(self, v: torch.Tensor) -> torch.Tensor:
        v_np = v.detach().cpu().numpy().ravel()
        u_np = self.ref_interpolator(v_np).T
        return torch.tensor(u_np, dtype=torch.float32, device=self.device)
