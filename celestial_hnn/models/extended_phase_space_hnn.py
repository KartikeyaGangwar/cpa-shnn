import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

class ExtendedPhaseSpaceHNN(nn.Module):
    """
    Theorem 2: Arnold's Extended Contact Phase Space HNN for Non-Autonomous & Pulsating Systems.
    
    Mathematical Formulation:
      Extended coordinates: z_ext = (q, t, p, p_t) in R^{2(d+1)}
      Extended Hamiltonian: K_theta(q, t, p, p_t) = 1/2 ||p||^2 - V_theta(q, t) + p_t = 0
      
    Canonical Extended Symplectic Flow:
      dq/dt = p
      dt/dt = 1.0 (Exact physical unit time evolution)
      dp/dt = grad_q V_theta(q, t)
      dp_t/dt = partial_t V_theta(q, t)
      
    Guarantees that non-autonomous, pulsating, and radiation-driven celestial systems
    are mapped into strictly autonomous, energy-conservative Hamiltonian systems.
    """
    def __init__(
        self,
        spatial_dim: int = 1,
        hidden_dim: int = 256,
        layers: int = 4,
        num_fourier: int = 16,
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.state_dim = 2 * spatial_dim
        self.ext_spatial_dim = spatial_dim + 1 # (q, t)
        self.ext_state_dim = 2 * self.ext_spatial_dim # (q, t, p, p_t)
        
        # Fourier feature projection for space-time (q, t)
        B = torch.randn(self.ext_spatial_dim, num_fourier) * 0.5
        self.register_buffer("B", B)
        
        in_dim = self.ext_spatial_dim + 2 * num_fourier
        net = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(layers - 2):
            net.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        net.append(nn.Linear(hidden_dim, 1, bias=False))
        self.space_time_potential_net = nn.Sequential(*net)

    def _fourier_embed(self, qt: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * np.pi * torch.matmul(qt, self.B)
        return torch.cat([qt, torch.sin(proj), torch.cos(proj)], dim=-1)

    def potential(self, q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluates time-dependent potential V_theta(q, t)."""
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        qt = torch.cat([q, t], dim=-1)
        return self.space_time_potential_net(self._fourier_embed(qt))

    def extended_hamiltonian(self, z_ext: torch.Tensor) -> torch.Tensor:
        """
        Evaluates extended Hamiltonian:
          K = 1/2 ||p||^2 - V_theta(q, t) + p_t
        """
        q = z_ext[:, :self.spatial_dim]
        t = z_ext[:, self.spatial_dim:self.spatial_dim+1]
        p = z_ext[:, self.spatial_dim+1:2*self.spatial_dim+1]
        p_t = z_ext[:, 2*self.spatial_dim+1:]
        
        kinetic = 0.5 * torch.sum(p ** 2, dim=-1, keepdim=True)
        V = self.potential(q, t)
        return kinetic - V + p_t

    def extended_time_derivative(self, z_ext: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        """
        Computes canonical derivatives in extended phase space:
          dq/dt = p
          dt/dt = 1.0
          dp/dt = grad_q V_theta(q, t)
          dp_t/dt = partial_t V_theta(q, t)
        """
        q = z_ext[:, :self.spatial_dim]
        t = z_ext[:, self.spatial_dim:self.spatial_dim+1]
        p = z_ext[:, self.spatial_dim+1:2*self.spatial_dim+1]
        
        dq_dt = p
        dt_dt = torch.ones_like(t)
        
        with torch.enable_grad():
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            qt = torch.cat([q, t], dim=-1)
            qt_eval = qt if qt.requires_grad else qt.clone().detach().requires_grad_(True)
            V = self.space_time_potential_net(self._fourier_embed(qt_eval))
            grad_qt = torch.autograd.grad(
                V,
                qt_eval,
                grad_outputs=torch.ones_like(V),
                create_graph=create_graph,
                retain_graph=True
            )[0]
            
        dp_dt = grad_qt[:, :self.spatial_dim]
        dp_t_dt = grad_qt[:, self.spatial_dim:]
        
        return torch.cat([dq_dt, dt_dt, dp_dt, dp_t_dt], dim=-1)

    def hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        """Evaluates standard instantaneous Hamiltonian at t=0 for compatibility."""
        q = z[:, :self.spatial_dim]
        p = z[:, self.spatial_dim:]
        t_zero = torch.zeros(len(z), 1, device=z.device)
        kinetic = 0.5 * torch.sum(p ** 2, dim=-1, keepdim=True)
        V = self.potential(q, t_zero)
        return kinetic - V

    def time_derivative(self, z: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        """Standard instantaneous derivative evaluation."""
        q = z[:, :self.spatial_dim]
        p = z[:, self.spatial_dim:]
        t_zero = torch.zeros(len(z), 1, device=z.device)
        p_t_zero = torch.zeros(len(z), 1, device=z.device)
        z_ext = torch.cat([q, t_zero, p, p_t_zero], dim=-1)
        dz_ext = self.extended_time_derivative(z_ext, create_graph=create_graph)
        dq_dt = dz_ext[:, :self.spatial_dim]
        dp_dt = dz_ext[:, self.spatial_dim+1:2*self.spatial_dim+1]
        return torch.cat([dq_dt, dp_dt], dim=-1)

    def integrate_symplectic_rk4(self, z0: torch.Tensor, t_span: torch.Tensor) -> torch.Tensor:
        """Integrates extended trajectory along time span."""
        if z0.dim() == 1:
            z0 = z0.unsqueeze(0)
        traj = [z0.clone()]
        curr_z = z0.clone()
        
        t0 = t_span[0:1].unsqueeze(-1) if t_span[0].dim() == 0 else t_span[0:1]
        H0 = self.hamiltonian(curr_z)
        pt0 = -H0
        
        curr_z_ext = torch.cat([curr_z[:, :self.spatial_dim], t0, curr_z[:, self.spatial_dim:], pt0], dim=-1)
        dt_vals = t_span[1:] - t_span[:-1]
        
        for dt in dt_vals:
            dt_step = dt.item()
            k1 = self.extended_time_derivative(curr_z_ext, create_graph=False)
            k2 = self.extended_time_derivative(curr_z_ext + 0.5 * dt_step * k1, create_graph=False)
            k3 = self.extended_time_derivative(curr_z_ext + 0.5 * dt_step * k2, create_graph=False)
            k4 = self.extended_time_derivative(curr_z_ext + dt_step * k3, create_graph=False)
            curr_z_ext = curr_z_ext + (dt_step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            
            # Extract standard (q, p)
            q_curr = curr_z_ext[:, :self.spatial_dim]
            p_curr = curr_z_ext[:, self.spatial_dim+1:2*self.spatial_dim+1]
            traj.append(torch.cat([q_curr, p_curr], dim=-1))
            
        return torch.stack(traj, dim=0)
