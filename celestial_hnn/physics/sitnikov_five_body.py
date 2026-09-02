import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List, Dict
from scipy.integrate import solve_ivp
from .base_hamiltonian import BaseHamiltonianSystem

class SitnikovFiveBodyHamiltonianSystem(BaseHamiltonianSystem):
    """
    System III: Elliptic Sitnikov Five-Body Problem in Canonical Phase Space
    Reference: Ullah, M. S., Idrisi, M. J., Kumar, V. (New Astronomy, 2020: 101398)
    """
    def __init__(
        self,
        eccentricity: float = 0.0,
        radiation_q: float = 0.85,
        V_max: float = 3.14159,
        z_init: Tuple[float, float] = (0.40, 0.0),
        device: Optional[torch.device] = None,
    ):
        self.e = eccentricity
        self.q = radiation_q
        self.z_init = z_init
        self.r0_sq = 0.5 * ((1.0 - self.e ** 2) ** 2)
        
        z0, vz0 = z_init
        z0_t = torch.tensor([z0, vz0], dtype=torch.float32)
        
        super().__init__(
            name="SitnikovFiveBodyHamiltonianSystem",
            spatial_dim=1,
            bounds_q=[(-1.5, 1.5)],
            bounds_p=[(-1.5, 1.5)],
            z0=z0_t,
            T_max=V_max,
            device=device,
        )
        self._precompute_reference_solution()

    def orbital_radius(self, v: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.e ** 2) / (1.0 + self.e * torch.cos(v))

    def exact_hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        pos_z = z[:, 0:1]
        pz = z[:, 1:2]
        potential = -4.0 * self.q / torch.sqrt(pos_z ** 2 + self.r0_sq)
        return 0.5 * (pz ** 2) + potential

    def canonical_derivatives(self, z: torch.Tensor) -> torch.Tensor:
        pos_z = z[:, 0:1]
        pz = z[:, 1:2]
        dz_dt = pz
        force_z = -4.0 * self.q * pos_z / ((pos_z ** 2 + self.r0_sq) ** 1.5)
        dpz_dt = force_z
        return torch.cat([dz_dt, dpz_dt], dim=-1)

    def _ode_rhs(self, v: float, z_np: np.ndarray) -> np.ndarray:
        pos_z, pz = z_np
        denom_prim = 1.0 + self.e * np.cos(v)
        r_v = (1.0 - self.e ** 2) / denom_prim
        denom_force = (pos_z ** 2 + 0.5 * (r_v ** 2)) ** 1.5
        grav_rad_force = 4.0 * self.q * pos_z / denom_force
        dz_dv = pz
        dpz_dv = (2.0 * self.e * np.sin(v) * pz - grav_rad_force) / denom_prim
        return [dz_dv, dpz_dv]

    def _precompute_reference_solution(self):
        sol = solve_ivp(
            self._ode_rhs,
            (0.0, self.T_max),
            list(self.z_init),
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
