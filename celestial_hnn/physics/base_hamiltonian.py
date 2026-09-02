import abc
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List, Union

class BaseHamiltonianSystem(abc.ABC):
    """
    Abstract Base Class for Canonical Hamiltonian Celestial Systems in Synodic/Inertial Frames.
    Phase Space Coordinates: z = (q, p) in R^{2d}
    Canonical Symplectic Form: omega = sum dq_i ^ dp_i
    """
    def __init__(
        self,
        name: str,
        spatial_dim: int,
        bounds_q: List[Tuple[float, float]],
        bounds_p: List[Tuple[float, float]],
        z0: torch.Tensor,
        T_max: float = 6.0,
        device: Optional[torch.device] = None,
    ):
        self.name = name
        self.spatial_dim = spatial_dim
        self.state_dim = 2 * spatial_dim
        self.bounds_q = bounds_q
        self.bounds_p = bounds_p
        self.T_max = T_max
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.z0 = z0.clone().detach().to(dtype=torch.float32, device=self.device).reshape(1, self.state_dim)

    @abc.abstractmethod
    def exact_hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        """Exact analytical Hamiltonian energy H(q, p)."""
        pass

    @abc.abstractmethod
    def canonical_derivatives(self, z: torch.Tensor) -> torch.Tensor:
        """Exact analytical time derivatives dz/dt = (dq/dt, dp/dt)."""
        pass

    def sample_phase_space(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples random phase space points z = (q, p) and exact canonical time derivatives dz/dt."""
        q_samples = []
        for b in self.bounds_q:
            q_i = (b[1] - b[0]) * torch.rand(n_samples, 1, device=self.device) + b[0]
            q_samples.append(q_i)
        p_samples = []
        for b in self.bounds_p:
            p_i = (b[1] - b[0]) * torch.rand(n_samples, 1, device=self.device) + b[0]
            p_samples.append(p_i)
            
        z = torch.cat(q_samples + p_samples, dim=-1)
        dz_dt = self.canonical_derivatives(z)
        return z, dz_dt

    @abc.abstractmethod
    def ground_truth_trajectory(self, t_span: torch.Tensor) -> torch.Tensor:
        """High-precision reference trajectory from DOP853 symplectic integrator."""
        pass

    def compute_trajectory_error(self, model: nn.Module, t_span: Optional[torch.Tensor] = None, n_points: int = 1000) -> float:
        """Computes relative L2 trajectory error: ||z_pred - z_exact||_2 / ||z_exact||_2."""
        if t_span is None:
            t_span = torch.linspace(0, self.T_max, n_points, device=self.device)
            
        z_exact = self.ground_truth_trajectory(t_span) # (T, 2d)
        
        with torch.no_grad():
            z_pred = model.integrate_symplectic_rk4(self.z0.squeeze(0), t_span).squeeze(1) # (T, 2d)
            l2_err = torch.norm(z_pred - z_exact, p=2)
            l2_ref = torch.norm(z_exact, p=2).clamp_min(1e-8)
            rel_err = (l2_err / l2_ref).item()
            
        return rel_err
