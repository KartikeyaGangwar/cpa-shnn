import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

class StructuredSeparableHNN(nn.Module):
    """
    Theorem 1: Structured Separable Hamiltonian Neural Network (Clean Smooth MLP, No Fourier).
    Decomposes: H(q, p) = 1/2 ||p||^2 + n(px*y - py*x) - V_theta(q)
    """
    def __init__(self, spatial_dim: int = 2, n_coriolis: float = 1.0, hidden_dim: int = 256, layers: int = 4):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.state_dim = 2 * spatial_dim
        self.n_coriolis = n_coriolis
        
        net = [nn.Linear(spatial_dim, hidden_dim), nn.Tanh()]
        for _ in range(layers - 2):
            net.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        net.append(nn.Linear(hidden_dim, 1, bias=False))
        self.potential_net = nn.Sequential(*net)
        
        for m in self.potential_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def potential(self, q: torch.Tensor) -> torch.Tensor:
        return self.potential_net(q)

    def hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        q = z[:, :self.spatial_dim]
        p = z[:, self.spatial_dim:]
        kinetic = 0.5 * torch.sum(p ** 2, dim=-1, keepdim=True)
        if self.spatial_dim == 2 and self.n_coriolis != 0.0:
            coriolis = self.n_coriolis * (p[:, 0:1] * q[:, 1:2] - p[:, 1:2] * q[:, 0:1])
        else:
            coriolis = 0.0
        return kinetic + coriolis - self.potential(q)

    def time_derivative(self, z: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        q = z[:, :self.spatial_dim]
        p = z[:, self.spatial_dim:]
        if self.spatial_dim == 2 and self.n_coriolis != 0.0:
            dq_dt = torch.cat([p[:, 0:1] + self.n_coriolis * q[:, 1:2], p[:, 1:2] - self.n_coriolis * q[:, 0:1]], dim=-1)
        else:
            dq_dt = p
        with torch.enable_grad():
            q_eval = q if q.requires_grad else q.clone().detach().requires_grad_(True)
            V = self.potential(q_eval)
            grad_V = torch.autograd.grad(V, q_eval, grad_outputs=torch.ones_like(V), create_graph=create_graph, retain_graph=True)[0]
        if self.spatial_dim == 2 and self.n_coriolis != 0.0:
            coriolis_force = torch.cat([self.n_coriolis * p[:, 1:2], -self.n_coriolis * p[:, 0:1]], dim=-1)
            dp_dt = coriolis_force + grad_V
        else:
            dp_dt = grad_V
        return torch.cat([dq_dt, dp_dt], dim=-1)

    def integrate_symplectic_rk4(self, z0: torch.Tensor, t_span: torch.Tensor) -> torch.Tensor:
        if z0.dim() == 1:
            z0 = z0.unsqueeze(0)
        traj = [z0.clone()]
        curr_z = z0.clone()
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
