import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

class SeparableGeneratingMapHNN(nn.Module):
    """
    Combo 1+3: Separable Neural Symplectic Generating Map.
    Combines Theorem 1 (Analytic Kinetic-Coriolis Separation) + Theorem 3 (Poincaré Generating Function).
    
    Formula:
      S_theta(q_k, p_{k+1}) = q_k . p_{k+1} + Delta_t * [ 1/2 ||p_{k+1}||^2 + n(px_{k+1}*y_k - py_{k+1}*x_k) - V_theta(q_k) ]
      
    Exact Symplectic Update:
      p_k = p_{k+1} + Delta_t * [ n * (py_{k+1}, -px_{k+1}) - grad_q V_theta(q_k) ]
      q_{k+1} = q_k + Delta_t * [ p_{k+1} + n * (y_k, -x_k) ]
    """
    def __init__(self, spatial_dim: int = 2, n_coriolis: float = 1.0, hidden_dim: int = 256, layers: int = 4, num_fourier: int = 16):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.state_dim = 2 * spatial_dim
        self.n_coriolis = n_coriolis
        
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
        return self.potential_net(self._fourier_embed(q))

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

    def step(self, q_k: torch.Tensor, p_k: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.enable_grad():
            q_eval = q_k if q_k.requires_grad else q_k.clone().detach().requires_grad_(True)
            V = self.potential(q_eval)
            grad_V = torch.autograd.grad(V, q_eval, grad_outputs=torch.ones_like(V), create_graph=False, retain_graph=True)[0]
        if self.spatial_dim == 2 and self.n_coriolis != 0.0:
            denom = 1.0 + (dt * self.n_coriolis) ** 2
            rhs_x = p_k[:, 0:1] + dt * grad_V[:, 0:1]
            rhs_y = p_k[:, 1:2] + dt * grad_V[:, 1:2]
            p_next_x = (rhs_x + dt * self.n_coriolis * rhs_y) / denom
            p_next_y = (rhs_y - dt * self.n_coriolis * rhs_x) / denom
            p_next = torch.cat([p_next_x, p_next_y], dim=-1)
            q_next_x = q_k[:, 0:1] + dt * (p_next_x + self.n_coriolis * q_k[:, 1:2])
            q_next_y = q_k[:, 1:2] + dt * (p_next_y - self.n_coriolis * q_k[:, 0:1])
            q_next = torch.cat([q_next_x, q_next_y], dim=-1)
        else:
            p_next = p_k + dt * grad_V
            q_next = q_k + dt * p_next
        return q_next, p_next

    def integrate_symplectic_rk4(self, z0: torch.Tensor, t_span: torch.Tensor) -> torch.Tensor:
        if z0.dim() == 1:
            z0 = z0.unsqueeze(0)
        traj = [z0.clone()]
        curr_q = z0[:, :self.spatial_dim].clone()
        curr_p = z0[:, self.spatial_dim:].clone()
        dt_vals = t_span[1:] - t_span[:-1]
        for dt in dt_vals:
            curr_q, curr_p = self.step(curr_q, curr_p, dt=dt.item())
            traj.append(torch.cat([curr_q, curr_p], dim=-1))
        return torch.stack(traj, dim=0)
