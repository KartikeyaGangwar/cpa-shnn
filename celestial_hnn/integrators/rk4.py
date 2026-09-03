import torch
from typing import Callable

def symplectic_rk4_integrate(
    vector_field_fn: Callable[[torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    t_span: torch.Tensor
) -> torch.Tensor:
    """
    Classical 4th-Order Runge-Kutta Integrator for Hamiltonian Vector Fields.
    """
    if z0.dim() == 1:
        z0 = z0.unsqueeze(0)
    traj = [z0.clone()]
    curr_z = z0.clone()
    dt_vals = t_span[1:] - t_span[:-1]
    for dt in dt_vals:
        dt_val = dt.item()
        k1 = vector_field_fn(curr_z)
        k2 = vector_field_fn(curr_z + 0.5 * dt_val * k1)
        k3 = vector_field_fn(curr_z + 0.5 * dt_val * k2)
        k4 = vector_field_fn(curr_z + dt_val * k3)
        curr_z = curr_z + (dt_val / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        traj.append(curr_z.clone())
    return torch.stack(traj, dim=0)
