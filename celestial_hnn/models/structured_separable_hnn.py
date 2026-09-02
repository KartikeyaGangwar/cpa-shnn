import torch
import torch.nn as nn
import numpy as np
from typing import Optional

class StructuredSeparableHNN(nn.Module):
    """
    Structured Separable Symplectic Hamiltonian Neural Network.
    
    Mathematical Decomposition:
      H(q, p) = 1/2 ||p||^2 + n (px * y - py * x) - V_theta(q)
      
    Exact Symplectic Canonical Equations:
      dq/dt = dH/dp = p + n * [-y, x]^T   (Analytically exact, zero autograd overhead)
      dp/dt = -dH/dq = n * [py, -px]^T + grad_q V_theta(q)
      
    Dimensionality Reduction:
      Input to neural network is strictly spatial coordinate q in R^d (2D or 1D),
      collapsing the parameter search manifold from 4D to 2D.
    """
    def __init__(
        self,
        spatial_dim: int = 2,
        n_coriolis: float = 1.0,
        hidden_dim: int = 256,
        layers: int = 4,
        num_fourier: int = 16,
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.state_dim = 2 * spatial_dim
        self.n_coriolis = n_coriolis
        self.num_fourier = num_fourier
        
        # Fourier Feature projection for spatial coordinates
        B = torch.randn(spatial_dim, num_fourier) * 0.5
        self.register_buffer("B", B)
        
        in_dim = spatial_dim + 2 * num_fourier
        net = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(layers - 2):
            net.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        net.append(nn.Linear(hidden_dim, 1, bias=False))
        self.potential_net = nn.Sequential(*net)

    def _fourier_embed(self, q: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * np.pi * torch.matmul(q, self.B)
        return torch.cat([q, torch.sin(proj), torch.cos(proj)], dim=-1)

    def potential(self, q: torch.Tensor) -> torch.Tensor:
        """Evaluates spatial scalar gravitational potential V_theta(q)."""
        return self.potential_net(self._fourier_embed(q))

    def hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        """
        Evaluates total Hamiltonian energy:
          H = 1/2 ||p||^2 + n(px * y - py * x) - V_theta(q)
        """
        q = z[:, :self.spatial_dim]
        p = z[:, self.spatial_dim:]
        
        # Exact analytic kinetic energy
        kinetic = 0.5 * torch.sum(p ** 2, dim=-1, keepdim=True)
        
        # Exact analytic Coriolis coupling
        if self.spatial_dim == 2 and self.n_coriolis != 0.0:
            x = q[:, 0:1]
            y = q[:, 1:2]
            px = p[:, 0:1]
            py = p[:, 1:2]
            coriolis = self.n_coriolis * (px * y - py * x)
        else:
            coriolis = 0.0
            
        V = self.potential(q)
        return kinetic + coriolis - V

    def time_derivative(self, z: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        """
        Computes exact canonical symplectic time derivatives:
          dq/dt = dH/dp (Analytic)
          dp/dt = -dH/dq (Autograd on V_theta + analytic Coriolis)
        """
        q = z[:, :self.spatial_dim]
        p = z[:, self.spatial_dim:]
        
        # 1. Analytic dq/dt
        if self.spatial_dim == 2 and self.n_coriolis != 0.0:
            x = q[:, 0:1]
            y = q[:, 1:2]
            px = p[:, 0:1]
            py = p[:, 1:2]
            dq_dt = torch.cat([px + self.n_coriolis * y, py - self.n_coriolis * x], dim=-1)
        else:
            dq_dt = p
            
        # 2. Analytic + Neural dp/dt
        with torch.enable_grad():
            q_eval = q if q.requires_grad else q.clone().detach().requires_grad_(True)
            V = self.potential(q_eval)
            grad_V = torch.autograd.grad(
                V,
                q_eval,
                grad_outputs=torch.ones_like(V),
                create_graph=create_graph,
                retain_graph=True
            )[0]
            
        if self.spatial_dim == 2 and self.n_coriolis != 0.0:
            dpx_coriolis = self.n_coriolis * py
            dpy_coriolis = -self.n_coriolis * px
            coriolis_force = torch.cat([dpx_coriolis, dpy_coriolis], dim=-1)
            dp_dt = coriolis_force + grad_V
        else:
            dp_dt = grad_V
            
        return torch.cat([dq_dt, dp_dt], dim=-1)

    def integrate_symplectic_rk4(self, z0: torch.Tensor, t_span: torch.Tensor) -> torch.Tensor:
        """Integrates phase space trajectory using 4th-order Runge-Kutta."""
        if z0.dim() == 1:
            z0 = z0.unsqueeze(0)
        traj = [z0.clone()]
        curr_z = z0.clone()
        dt_vals = t_span[1:] - t_span[:-1]
        for dt in dt_vals:
            dt_step = dt.item()
            k1 = self.time_derivative(curr_z, create_graph=False)
            k2 = self.time_derivative(curr_z + 0.5 * dt_step * k1, create_graph=False)
            k3 = self.time_derivative(curr_z + 0.5 * dt_step * k2, create_graph=False)
            k4 = self.time_derivative(curr_z + dt_step * k3, create_graph=False)
            curr_z = curr_z + (dt_step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            traj.append(curr_z.clone())
        return torch.stack(traj, dim=0)
