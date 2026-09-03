import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

class SeparableExtendedContactHNN(nn.Module):
    """
    Non-Autonomous Geometric Architecture (Clean Smooth MLP, No Fourier):
    Unifies Theorem 1 (Separable Kinetic-Coriolis) + Theorem 2 (Arnold Contact Space).
    """
    def __init__(
        self,
        spatial_dim: int = 2,
        n_coriolis: float = 1.0,
        hidden_dim: int = 256,
        layers: int = 4
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.ext_spatial_dim = spatial_dim + 1 # (q, t)
        self.state_dim = 2 * self.ext_spatial_dim # (q, t, p, pt)
        self.n_coriolis = n_coriolis
        
        # Clean smooth multi-layer perceptron (No high-frequency Fourier multiplier)
        net = [nn.Linear(self.ext_spatial_dim, hidden_dim), nn.Tanh()]
        for _ in range(layers - 2):
            net.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        net.append(nn.Linear(hidden_dim, 1, bias=False))
        self.potential_net = nn.Sequential(*net)
        
        # Xavier initialization for smooth gradients
        for m in self.potential_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def potential(self, q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        qt = torch.cat([q, t], dim=-1)
        return self.potential_net(qt)

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
            
        # 2. dt/dt = 1.0 (Exact Arnold Clock)
        dt_dt = torch.ones_like(t)
        
        # 3. Autograd on smooth potential V_theta(q, t)
        with torch.enable_grad():
            qt = torch.cat([q, t], dim=-1)
            qt_eval = qt if qt.requires_grad else qt.clone().detach().requires_grad_(True)
            V = self.potential_net(qt_eval)
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
