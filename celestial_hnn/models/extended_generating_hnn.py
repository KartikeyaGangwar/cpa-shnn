import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

class ExtendedGeneratingMapHNN(nn.Module):
    """
    Combo 2+3: Extended Space Neural Symplectic Generating Map with Fourier embeddings.
    """
    def __init__(
        self,
        spatial_dim: int = 1,
        hidden_dim: int = 256,
        layers: int = 4,
        use_fourier: bool = True,
        num_fourier: int = 16,
        fourier_scale: float = 0.5
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.ext_dim = spatial_dim + 1
        self.state_dim = 2 * self.ext_dim
        self.use_fourier = use_fourier
        
        if use_fourier:
            B = torch.randn(self.ext_dim, num_fourier) * fourier_scale
            self.register_buffer("B", B)
            in_dim = self.ext_dim + 2 * num_fourier
        else:
            in_dim = self.ext_dim
            
        net = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(layers - 2):
            net.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        net.append(nn.Linear(hidden_dim, 1, bias=False))
        self.potential_net = nn.Sequential(*net)
        
        for m in self.potential_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def _embed(self, qt: torch.Tensor) -> torch.Tensor:
        if self.use_fourier:
            proj = 2.0 * np.pi * torch.matmul(qt, self.B)
            return torch.cat([qt, torch.sin(proj), torch.cos(proj)], dim=-1)
        return qt

    def potential(self, q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        qt = torch.cat([q, t], dim=-1)
        return self.potential_net(self._embed(qt))

    def time_derivative(self, z_ext: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        q = z_ext[:, :self.spatial_dim]
        t = z_ext[:, self.spatial_dim:self.spatial_dim+1]
        p = z_ext[:, self.spatial_dim+1:2*self.spatial_dim+1]
        pt = z_ext[:, 2*self.spatial_dim+1:]
        
        dq_dt = p
        dt_dt = torch.ones_like(t)
        
        with torch.enable_grad():
            qt = torch.cat([q, t], dim=-1)
            qt_eval = qt if qt.requires_grad else qt.clone().detach().requires_grad_(True)
            V = self.potential(qt_eval[:, :self.spatial_dim], qt_eval[:, self.spatial_dim:])
            grad_qt = torch.autograd.grad(V, qt_eval, grad_outputs=torch.ones_like(V), create_graph=create_graph, retain_graph=True)[0]
            
        grad_q = grad_qt[:, :self.spatial_dim]
        grad_t = grad_qt[:, self.spatial_dim:]
        
        dp_dt = -grad_q
        dpt_dt = -grad_t
        return torch.cat([dq_dt, dt_dt, dp_dt, dpt_dt], dim=-1)

    def step(self, q_k: torch.Tensor, p_k: torch.Tensor, t_k: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.enable_grad():
            if t_k.dim() == 1:
                t_k = t_k.unsqueeze(-1)
            qt = torch.cat([q_k, t_k], dim=-1)
            qt_eval = qt if qt.requires_grad else qt.clone().detach().requires_grad_(True)
            V = self.potential(qt_eval[:, :self.spatial_dim], qt_eval[:, self.spatial_dim:])
            grad_qt = torch.autograd.grad(V, qt_eval, grad_outputs=torch.ones_like(V), create_graph=False, retain_graph=True)[0]
        grad_q = grad_qt[:, :self.spatial_dim]
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
