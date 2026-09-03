import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

class NeuralSymplecticGeneratingMap(nn.Module):
    """
    Theorem 3: Neural Symplectic Poincaré-Jacobi Generating Map (GFNN).
    Directly parameterizes the Type-I Generating Function S_theta(q_k, p_{k+1}).
    """
    def __init__(
        self,
        spatial_dim: int = 2,
        hidden_dim: int = 256,
        layers: int = 4,
        use_fourier: bool = True,
        num_fourier: int = 16,
        fourier_scale: float = 0.5
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.state_dim = 2 * spatial_dim
        self.use_fourier = use_fourier
        
        if use_fourier:
            B = torch.randn(self.state_dim, num_fourier) * fourier_scale
            self.register_buffer("B", B)
            in_dim = self.state_dim + 2 * num_fourier
        else:
            in_dim = self.state_dim
            
        net = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(layers - 2):
            net.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        net.append(nn.Linear(hidden_dim, 1, bias=False))
        self.g_net = nn.Sequential(*net)
        
        for m in self.g_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def _embed(self, z: torch.Tensor) -> torch.Tensor:
        if self.use_fourier:
            proj = 2.0 * np.pi * torch.matmul(z, self.B)
            return torch.cat([z, torch.sin(proj), torch.cos(proj)], dim=-1)
        return z

    def generating_function(self, q_k: torch.Tensor, p_next: torch.Tensor) -> torch.Tensor:
        qp = torch.cat([q_k, p_next], dim=-1)
        return self.g_net(self._embed(qp))

    def time_derivative(self, z: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        q = z[:, :self.spatial_dim]
        p = z[:, self.spatial_dim:]
        with torch.enable_grad():
            z_eval = z if z.requires_grad else z.clone().detach().requires_grad_(True)
            G = self.generating_function(z_eval[:, :self.spatial_dim], z_eval[:, self.spatial_dim:])
            dG = torch.autograd.grad(G, z_eval, grad_outputs=torch.ones_like(G), create_graph=create_graph, retain_graph=True)[0]
        dq = dG[:, self.spatial_dim:]
        dp = -dG[:, :self.spatial_dim]
        return torch.cat([dq, dp], dim=-1)

    def hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        return self.generating_function(z[:, :self.spatial_dim], z[:, self.spatial_dim:])

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
