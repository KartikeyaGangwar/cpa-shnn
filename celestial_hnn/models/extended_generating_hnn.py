import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

class ExtendedGeneratingMapHNN(nn.Module):
    """
    Combo 2+3: Extended Space Neural Symplectic Generating Map.
    Combines Theorem 2 (Arnold Contact Space) + Theorem 3 (Generating Function).
    """
    def __init__(self, spatial_dim: int = 1, hidden_dim: int = 256, layers: int = 4, num_fourier: int = 16):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.ext_dim = spatial_dim + 1
        
        B = torch.randn(self.ext_dim, num_fourier) * 0.5
        self.register_buffer("B", B)
        in_dim = self.ext_dim + 2 * num_fourier
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

    def hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        q = z[:, :self.spatial_dim]
        p = z[:, self.spatial_dim:]
        t0 = torch.zeros(len(z), 1, device=z.device)
        return 0.5 * torch.sum(p ** 2, dim=-1, keepdim=True) - self.potential(q, t0)

    def time_derivative(self, z: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        q = z[:, :self.spatial_dim]
        p = z[:, self.spatial_dim:]
        t0 = torch.zeros(len(z), 1, device=z.device)
        with torch.enable_grad():
            qt = torch.cat([q, t0], dim=-1)
            qt_eval = qt if qt.requires_grad else qt.clone().detach().requires_grad_(True)
            V = self.potential_net(self._fourier_embed(qt_eval))
            grad_qt = torch.autograd.grad(V, qt_eval, grad_outputs=torch.ones_like(V), create_graph=create_graph, retain_graph=True)[0]
        dq_dt = p
        dp_dt = grad_qt[:, :self.spatial_dim]
        return torch.cat([dq_dt, dp_dt], dim=-1)

    def step(self, q_k: torch.Tensor, p_k: torch.Tensor, t_k: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.enable_grad():
            if t_k.dim() == 1:
                t_k = t_k.unsqueeze(-1)
            qt = torch.cat([q_k, t_k], dim=-1)
            qt_eval = qt if qt.requires_grad else qt.clone().detach().requires_grad_(True)
            V = self.potential_net(self._fourier_embed(qt_eval))
            grad_qt = torch.autograd.grad(V, qt_eval, grad_outputs=torch.ones_like(V), create_graph=False, retain_graph=True)[0]
        dp_dt = grad_qt[:, :self.spatial_dim]
        p_next = p_k + dt * dp_dt
        q_next = q_k + dt * p_next
        t_next = t_k + dt
        return q_next, p_next, t_next

    def integrate_symplectic_rk4(self, z0: torch.Tensor, t_span: torch.Tensor) -> torch.Tensor:
        if z0.dim() == 1:
            z0 = z0.unsqueeze(0)
        traj = [z0.clone()]
        curr_q = z0[:, :self.spatial_dim].clone()
        curr_p = z0[:, self.spatial_dim:].clone()
        curr_t = t_span[0:1].unsqueeze(-1) if t_span[0].dim() == 0 else t_span[0:1]
        dt_vals = t_span[1:] - t_span[:-1]
        for dt in dt_vals:
            curr_q, curr_p, curr_t = self.step(curr_q, curr_p, curr_t, dt=dt.item())
            traj.append(torch.cat([curr_q, curr_p], dim=-1))
        return torch.stack(traj, dim=0)
