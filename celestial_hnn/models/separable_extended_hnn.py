import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

class SeparableExtendedContactHNN(nn.Module):
    """
    Non-Autonomous Geometric Architecture with Fourier positional encodings:
    Unifies Theorem 1 (Separable Kinetic-Coriolis) + Theorem 2 (Arnold Contact Space).
    """
    def __init__(
        self,
        spatial_dim: int = 2,
        n_coriolis: float = 1.0,
        hidden_dim: int = 256,
        layers: int = 4,
        use_fourier: bool = True,
        num_fourier: int = 16,
        fourier_scale: float = 0.5
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.ext_spatial_dim = spatial_dim + 1 # (q, t)
        self.state_dim = 2 * self.ext_spatial_dim # (q, t, p, pt)
        self.n_coriolis = n_coriolis
        self.use_fourier = use_fourier
        
        if use_fourier:
            B = torch.randn(self.ext_spatial_dim, num_fourier) * fourier_scale
            self.register_buffer("B", B)
            in_dim = self.ext_spatial_dim + 2 * num_fourier
        else:
            in_dim = self.ext_spatial_dim
            
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
        
        if self.spatial_dim == 2 and self.n_coriolis != 0.0:
            dq_dt = torch.cat([p[:, 0:1] + self.n_coriolis * q[:, 1:2], p[:, 1:2] - self.n_coriolis * q[:, 0:1]], dim=-1)
        else:
            dq_dt = p
        dt_dt = torch.ones_like(t)
        
        with torch.enable_grad():
            qt = torch.cat([q, t], dim=-1)
            qt_eval = qt if qt.requires_grad else qt.clone().detach().requires_grad_(True)
            V = self.potential(qt_eval[:, :self.spatial_dim], qt_eval[:, self.spatial_dim:])
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

    def integrate_taylor_jet(self, z0_ext: torch.Tensor, t_span: torch.Tensor, order: int = 6) -> torch.Tensor:
        from celestial_hnn.integrators.taylor_jet import taylor_jet_integrate
        return taylor_jet_integrate(lambda z: self.time_derivative(z, create_graph=False), z0_ext, t_span, order=order)
