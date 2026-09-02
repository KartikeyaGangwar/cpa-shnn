import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List, Dict
from scipy.integrate import solve_ivp
from .base_hamiltonian import BaseHamiltonianSystem

class SitnikovFiveBodyHamiltonianSystem(BaseHamiltonianSystem):
    """
    System III: Sitnikov Five-Body Problem in Canonical Phase Space with Dual Regime.
    - Regime 'regular': Circular Sitnikov (e=0, T=3.14, clean oscillation).
    - Regime 'chaotic': Elliptic Non-Autonomous Sitnikov (e=0.15, T=6.28, pulsating potential).
    Reference: Ullah, M. S., Idrisi, M. J., Kumar, V. (New Astronomy, 2020: 101398)
    """
    def __init__(
        self,
        regime: str = "regular",
        radiation_q: float = 0.85,
        device: Optional[torch.device] = None,
    ):
        self.regime = regime
        self.q = radiation_q
        
        if regime == "regular":
            self.e = 0.0
            self.T_max = 3.14159
            self.z_init = (0.40, 0.0)
        else: # chaotic elliptic pulsation
            self.e = 0.15
            self.T_max = 6.283185
            self.z_init = (0.50, 0.0)
            
        self.r0_sq = 0.5 * ((1.0 - self.e ** 2) ** 2)
        z0, vz0 = self.z_init
        z0_t = torch.tensor([z0, vz0], dtype=torch.float32)
        
        super().__init__(
            name=f"SitnikovFiveBody_{regime}",
            spatial_dim=1,
            bounds_q=[(-1.5, 1.5)],
            bounds_p=[(-1.5, 1.5)],
            z0=z0_t,
            T_max=self.T_max,
            device=device,
        )
        self._precompute_reference_solution()

    def orbital_radius(self, v: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.e ** 2) / (1.0 + self.e * torch.cos(v))

    def exact_hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        pos_z = z[:, 0:1]
        pz = z[:, 1:2]
        kinetic = 0.5 * (pz ** 2)
        phi = 4.0 * self.q / torch.sqrt(pos_z ** 2 + self.r0_sq)
        return kinetic - phi

    def canonical_derivatives(self, z: torch.Tensor) -> torch.Tensor:
        pos_z = z[:, 0:1]
        pz = z[:, 1:2]
        dz_dt = pz
        denom = (pos_z ** 2 + self.r0_sq) ** 1.5
        dpz_dt = -4.0 * self.q * pos_z / denom
        return torch.cat([dz_dt, dpz_dt], dim=-1)

    def _ode_rhs(self, v: float, z_np: np.ndarray) -> np.ndarray:
        z, pz = z_np
        r_v = (1.0 - self.e**2) / (1.0 + self.e * np.cos(v))
        denom = (z**2 + r_v**2)**1.5
        dz_dv = pz
        dpz_dv = -4.0 * self.q * z / denom
        return [dz_dv, dpz_dv]

    def _precompute_reference_solution(self):
        z0, vz0 = self.z_init
        sol = solve_ivp(
            self._ode_rhs,
            (0.0, self.T_max),
            [z0, vz0],
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
