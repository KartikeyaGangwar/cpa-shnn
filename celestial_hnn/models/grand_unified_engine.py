import torch
import torch.nn as nn
from typing import Tuple, Optional

class GrandUnifiedSymplecticEngine(nn.Module):
    """
    Combo 1+2+3: Grand Unified Celestial Symplectic Engine (Clean Smooth MLP, No Fourier).
    """
    def __init__(self, spatial_dim: int = 2, n_coriolis: float = 1.0, hidden_dim: int = 256, layers: int = 4):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.ext_dim = spatial_dim + 1 # (q, t)
        self.state_dim = 2 * self.ext_dim
        self.n_coriolis = n_coriolis
        
        net = [nn.Linear(self.ext_dim, hidden_dim), nn.Tanh()]
        for _ in range(layers - 2):
            net.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        net.append(nn.Linear(hidden_dim, 1, bias=False))
        self.potential_net = nn.Sequential(*net)
        for m in self.potential_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def potential(self, q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        qt = torch.cat([q, t], dim=-1)
        return self.potential_net(qt)

    def time_derivative(self, z_ext: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        q = z_ext[:, :self.spatial_dim]
        t = z_ext[:, self.spatial_dim:self.spatial_dim+1]
        p = z_ext[:, self.spatial_dim+1:2*self.spatial_dim+1]
        pt = z_ext[:, 2*self.spatial_dim+1:]
        
        if self.spatial_dim == 2 and self.n_coriolis != 0.0:
            dq_dt = torch.cat([p[:, 0:1] + self.n_coriolis * q[:, 1:2], p[:, 1:2] - self.n_coriolis * q[:, 0:1]], dim=-1)
        else:
            dq_dt = p
        dt_dt = torch.ones_like(t)
        
        with torch.enable_grad():
            qt = torch.cat([q, t], dim=-1)
            qt_eval = qt if qt.requires_grad else qt.clone().detach().requires_grad_(True)
            V = self.potential_net(qt_eval)
            grad_qt = torch.autograd.grad(V, qt_eval, grad_outputs=torch.ones_like(V), create_graph=create_graph, retain_graph=True)[0]
        grad_q = grad_qt[:, :self.spatial_dim]
        grad_t = grad_qt[:, self.spatial_dim:]
        
        if self.spatial_dim == 2 and self.n_coriolis != 0.0:
            coriolis_force = torch.cat([self.n_coriolis * p[:, 1:2], -self.n_coriolis * p[:, 0:1]], dim=-1)
            dp_dt = coriolis_force - grad_q
        else:
            dp_dt = -grad_q
        dpt_dt = -grad_t
        return torch.cat([dq_dt, dt_dt, dp_dt, dpt_dt], dim=-1)

    def step(self, q_k: torch.Tensor, p_k: torch.Tensor, t_k: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.enable_grad():
            if t_k.dim() == 1:
                t_k = t_k.unsqueeze(-1)
            qt = torch.cat([q_k, t_k], dim=-1)
            qt_eval = qt if qt.requires_grad else qt.clone().detach().requires_grad_(True)
            V = self.potential_net(qt_eval)
            grad_qt = torch.autograd.grad(V, qt_eval, grad_outputs=torch.ones_like(V), create_graph=False, retain_graph=True)[0]
        grad_q = grad_qt[:, :self.spatial_dim]
        
        if self.spatial_dim == 2 and self.n_coriolis != 0.0:
            denom = 1.0 + (dt * self.n_coriolis) ** 2
            rhs_x = p_k[:, 0:1] - dt * grad_q[:, 0:1]
            rhs_y = p_k[:, 1:2] - dt * grad_q[:, 1:2]
            p_next_x = (rhs_x + dt * self.n_coriolis * rhs_y) / denom
            p_next_y = (rhs_y - dt * self.n_coriolis * rhs_x) / denom
            p_next = torch.cat([p_next_x, p_next_y], dim=-1)
            q_next_x = q_k[:, 0:1] + dt * (p_next_x + self.n_coriolis * q_k[:, 1:2])
            q_next_y = q_k[:, 1:2] + dt * (p_next_y - self.n_coriolis * q_k[:, 0:1])
            q_next = torch.cat([q_next_x, q_next_y], dim=-1)
        else:
            p_next = p_k - dt * grad_q
            q_next = q_k + dt * p_next
        t_next = t_k + dt
        return q_next, p_next, t_next

    def integrate_symplectic_rk4(self, z0_ext: torch.Tensor, t_span: torch.Tensor) -> torch.Tensor:
        if z0_ext.dim() == 1:
            z0_ext = z0_ext.unsqueeze(0)
        traj = [z0_ext.clone()]
        curr_q = z0_ext[:, :self.spatial_dim].clone()
        curr_t = z0_ext[:, self.spatial_dim:self.spatial_dim+1].clone()
        curr_p = z0_ext[:, self.spatial_dim+1:2*self.spatial_dim+1].clone()
        curr_pt = z0_ext[:, 2*self.spatial_dim+1:].clone()
        dt_vals = t_span[1:] - t_span[:-1]
        
        for dt in dt_vals:
            dt_val = dt.item()
            curr_q, curr_p, curr_t = self.step(curr_q, curr_p, curr_t, dt=dt_val)
            traj.append(torch.cat([curr_q, curr_t, curr_p, curr_pt], dim=-1))
        return torch.stack(traj, dim=0)
