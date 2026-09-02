import abc
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Union

class BaseCelestialSystem(abc.ABC):
    """
    Abstract Base Class for Celestial Mechanics and Chaotic Phase Space Systems.
    Pure Initial Value Problem (IVP) & Hamiltonian Dynamics Formulation.
    """
    def __init__(
        self,
        name: str,
        in_dim: int,
        out_dim: int,
        bounds: List[Tuple[float, float]],
        u0: torch.Tensor,
        device: Optional[torch.device] = None,
    ):
        self.name = name
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bounds = bounds
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.u0 = u0.clone().detach().to(dtype=torch.float32, device=self.device)

    @abc.abstractmethod
    def sample_interior(self, n_samples: int) -> torch.Tensor:
        """Samples collocation time points t in [0, T_max]."""
        pass

    def sample_initial(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples exact t=0 initial condition state."""
        t_0 = torch.zeros((n_samples, 1), dtype=torch.float32, device=self.device)
        u_0 = self.u0.repeat(n_samples, 1)
        return t_0, u_0

    def sample_boundary(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """In pure IVP, boundary is identical to initial condition t=0."""
        return self.sample_initial(n_samples)

    @abc.abstractmethod
    def compute_residuals(self, model: nn.Module, t: torch.Tensor) -> torch.Tensor:
        """Computes ODE / Hamiltonian residuals."""
        pass

    def compute_initial_loss(
        self, model: nn.Module, t_ic: torch.Tensor, u_ic_exact: torch.Tensor
    ) -> torch.Tensor:
        """Initial state Mean Squared Error."""
        u_pred = model(t_ic)
        return torch.mean((u_pred - u_ic_exact) ** 2)

    def compute_boundary_loss(
        self, model: nn.Module, t_bc: torch.Tensor, u_bc_exact: torch.Tensor
    ) -> torch.Tensor:
        """Boundary loss for IVP acts as auxiliary IC enforcement."""
        return self.compute_initial_loss(model, t_bc, u_bc_exact)

    def compute_energy_conservation_loss(self, model: nn.Module, t: torch.Tensor) -> torch.Tensor:
        """Default Hamiltonian / Jacobi energy conservation loss (overridden in systems)."""
        return torch.tensor(0.0, device=self.device)

    @abc.abstractmethod
    def exact_solution(self, t: torch.Tensor) -> torch.Tensor:
        """High-precision reference solution from DOP853 numerical integrator."""
        pass

    def compute_relative_l2_error(
        self, model: nn.Module, t_test: Optional[torch.Tensor] = None, n_test: int = 2000
    ) -> float:
        """Computes relative L2 error: ||u_pred - u_exact||_2 / ||u_exact||_2."""
        if t_test is None:
            t_test = self.sample_interior(n_test)
            
        with torch.no_grad():
            u_pred = model(t_test)
            u_exact = self.exact_solution(t_test)
            l2_err = torch.norm(u_pred - u_exact, p=2)
            l2_ref = torch.norm(u_exact, p=2).clamp_min(1e-8)
            rel_error = (l2_err / l2_ref).item()
            
        return rel_error
