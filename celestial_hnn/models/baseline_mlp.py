import torch
import torch.nn as nn

class BaselineVectorFieldMLP(nn.Module):
    """
    Standard Baseline Vector Field Multi-Layer Perceptron.
    Directly predicts state derivatives dz/dt = f_theta(z) without symplectic geometry.
    """
    def __init__(self, state_dim: int = 4, hidden_dim: int = 256, layers: int = 4):
        super().__init__()
        self.state_dim = state_dim
        net = [nn.Linear(state_dim, hidden_dim), nn.Tanh()]
        for _ in range(layers - 2):
            net.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        net.append(nn.Linear(hidden_dim, state_dim))
        self.net = nn.Sequential(*net)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def time_derivative(self, z: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        return self.forward(z)

    def integrate_symplectic_rk4(self, z0: torch.Tensor, t_span: torch.Tensor) -> torch.Tensor:
        if z0.dim() == 1:
            z0 = z0.unsqueeze(0)
        traj = [z0.clone()]
        curr_z = z0.clone()
        dt_vals = t_span[1:] - t_span[:-1]
        for dt in dt_vals:
            dt_step = dt.item()
            k1 = self.time_derivative(curr_z)
            k2 = self.time_derivative(curr_z + 0.5 * dt_step * k1)
            k3 = self.time_derivative(curr_z + 0.5 * dt_step * k2)
            k4 = self.time_derivative(curr_z + dt_step * k3)
            curr_z = curr_z + (dt_step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            traj.append(curr_z.clone())
        return torch.stack(traj, dim=0)

    def integrate_taylor_jet(self, z0: torch.Tensor, t_span: torch.Tensor, order: int = 6) -> torch.Tensor:
        from celestial_hnn.integrators.taylor_jet import taylor_jet_integrate
        return taylor_jet_integrate(lambda z: self(z), z0, t_span, order=order)
