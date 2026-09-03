import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

class SeparableExtendedContactHNN(nn.Module):
    """
    The Ultimate Non-Autonomous Geometric Architecture:
    Unification of:
      - Theorem 1: Separable Symplectic Kinetic-Coriolis Decomposition (p, px*y - py*x)
      - Theorem 2: Arnold's Extended Contact Phase Space (q, t, p, pt)
      
    Extended Canonical Hamiltonian:
      K_theta(q, t, p, pt) = 1/2 ||p||^2 + n(px*y - py*x) + V_theta(q, t) + pt = 0
      
    Exact Contact Symplectic Equations:
      dq/dt = p + n * (y, -x)       (Analytically exact)
      dt/dt = 1.0                   (Arnold's unit clock)
      dp/dt = n * (py, -px) - grad_q V_theta(q, t)
      dpt/dt = -∂V_theta/∂t         (Energy exchange rate)
    """
    def __init__(
        self,
        spatial_dim: int = 2,
        n_coriolis: float = 1.0,
        hidden_dim: int = 256,
        layers: int = 4,
        num_fourier: int = 16
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.ext_spatial_dim = spatial_dim + 1 # (q, t)
        self.state_dim = 2 * self.ext_spatial_dim # (q, t, p, pt)
        self.n_coriolis = n_coriolis
        
        # Fourier positional encoding for spatial-temporal manifold (q, t)
        B = torch.randn(self.ext_spatial_dim, num_fourier) * 0.5
        self.register_buffer("B", B)
        in_dim = self.ext_spatial_dim + 2 * num_fourier
        
        net = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(layers - 2):
            net.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        net.append(nn.Linear(hidden_dim, 1, bias=False))
        self.potential_net = nn.Sequential(*net)

    def _fourier_embed(self, qt: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * np.pi * torch.matmul(qt, self.B)
        return torch.cat([qt, torch.sin(proj), torch.cos(proj)], dim=-1)

    def potential(self, q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        qt = torch.cat([q, t], dim=-1)
        return self.potential_net(self._fourier_embed(qt))

    def extended_hamiltonian(self, z_ext: torch.Tensor) -> torch.Tensor:
        q = z_ext[:, :self.spatial_dim]
        t = z_ext[:, self.spatial_dim:self.spatial_dim+1]
        p = z_ext[:, self.spatial_dim+1:2*self.spatial_dim+1]
        pt = z_ext[:, 2*self.spatial_dim+1:]
        
        kinetic = 0.5 * torch.sum(p ** 2, dim=-1, keepdim=True)
        if self.spatial_dim == 2 and self.n_coriolis != 0.0:
            coriolis = self.n_coriolis * (p[:, 0:1] * q[:, 1:2] - p[:, 1:2] * q[:, 0:1])
        else:
            coriolis = 0.0
        V = self.potential(q, t)
        return kinetic + coriolis + V + pt

    def time_derivative(self, z_ext: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        q = z_ext[:, :self.spatial_dim]
        t = z_ext[:, self.spatial_dim:self.spatial_dim+1]
        p = z_ext[:, self.spatial_dim+1:2*self.spatial_dim+1]
        pt = z_ext[:, 2*self.spatial_dim+1:]
        
        # 1. dq/dt
        if self.spatial_dim == 2 and self.n_coriolis != 0.0:
            dq_dt = torch.cat([p[:, 0:1] + self.n_coriolis * q[:, 1:2], p[:, 1:2] - self.n_coriolis * q[:, 0:1]], dim=-1)
        else:
            dq_dt = p
            
        # 2. dt/dt = 1.0
        dt_dt = torch.ones_like(t)
        
        # 3. Autograd for spatial and temporal gradients of V_theta(q, t)
        with torch.enable_grad():
            qt = torch.cat([q, t], dim=-1)
            qt_eval = qt if qt.requires_grad else qt.clone().detach().requires_grad_(True)
            V = self.potential_net(self._fourier_embed(qt_eval))
            grad_qt = torch.autograd.grad(V, qt_eval, grad_outputs=torch.ones_like(V), create_graph=create_graph, retain_graph=True)[0]
            
        grad_q = grad_qt[:, :self.spatial_dim]
        grad_t = grad_qt[:, self.spatial_dim:]
        
        # 4. dp/dt
        if self.spatial_dim == 2 and self.n_coriolis != 0.0:
            coriolis_force = torch.cat([self.n_coriolis * p[:, 1:2], -self.n_coriolis * p[:, 0:1]], dim=-1)
            dp_dt = coriolis_force - grad_q
        else:
            dp_dt = -grad_q
            
        # 5. dpt/dt = -∂V/∂t
        dpt_dt = -grad_t
        
        return torch.cat([dq_dt, dt_dt, dp_dt, dpt_dt], dim=-1)

    def integrate_symplectic_rk4(self, z0_ext: torch.Tensor, t_span: torch.Tensor) -> torch.Tensor:
        if z0_ext.dim() == 1:
            z0_ext = z0_ext.unsqueeze(0)
        traj = [z0_ext.clone()]
        curr_z = z0_ext.clone()
        dt_vals = t_span[1:] - t_span[:-1]
        
        for dt in dt_vals:
            dt_val = dt.item()
            k1 = self.time_derivative(curr_z, create_graph=False)
            k2 = self.time_derivative(curr_z + 0.5 * dt_val * k1, create_graph=False)
            k3 = self.time_derivative(curr_z + 0.5 * dt_val * k2, create_graph=False)
            k4 = self.time_derivative(curr_z + dt_val * k3, create_graph=False)
            curr_z = curr_z + (dt_val / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            traj.append(curr_z.clone())
            
        return torch.stack(traj, dim=0)
