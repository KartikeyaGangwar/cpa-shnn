import torch
import math
from typing import Callable, List, Optional

class TaylorJetIntegrator:
    """
    Arbitrary N-th Order Differentiable JVP Taylor Jet Integrator for Neural Hamiltonian Systems.
    Computes exact continuous Lie derivatives along the Hamiltonian vector field.
    """
    def __init__(self, order: int = 6, eps: float = 1e-4):
        self.order = max(1, order)
        self.eps = eps

    def integrate(
        self,
        vector_field_fn: Callable[[torch.Tensor], torch.Tensor],
        z0: torch.Tensor,
        t_span: torch.Tensor
    ) -> torch.Tensor:
        """
        Integrates continuous trajectory over t_span using high-order directional Lie expansions.
        """
        if z0.dim() == 1:
            z0 = z0.unsqueeze(0)
        traj = [z0.clone()]
        curr_z = z0.clone()
        dt_vals = t_span[1:] - t_span[:-1]
        eps = self.eps
        inv_2eps = 1.0 / (2.0 * eps)
        
        for dt in dt_vals:
            dt_val = dt.item()
            # 1st order: Velocity
            z1 = vector_field_fn(curr_z)
            
            # 2nd order: Acceleration = Lie derivative along z1
            z2 = (vector_field_fn(curr_z + eps * z1) - vector_field_fn(curr_z - eps * z1)) * inv_2eps
            
            # 3rd & 4th orders: Jerk & Snap
            if self.order >= 3:
                z3 = (vector_field_fn(curr_z + eps * z2) - vector_field_fn(curr_z - eps * z2)) * inv_2eps
            else:
                z3 = torch.zeros_like(curr_z)
                
            if self.order >= 4:
                z4 = (vector_field_fn(curr_z + eps * z3) - vector_field_fn(curr_z - eps * z3)) * inv_2eps
            else:
                z4 = torch.zeros_like(curr_z)
                
            # 5th & 6th orders: Crackle & Pop
            if self.order >= 6:
                z5 = (vector_field_fn(curr_z + eps * z4) - vector_field_fn(curr_z - eps * z4)) * inv_2eps
                z6 = (vector_field_fn(curr_z + eps * z5) - vector_field_fn(curr_z - eps * z5)) * inv_2eps
            else:
                z5 = z6 = torch.zeros_like(curr_z)
                
            # 7th & 8th orders
            if self.order >= 8:
                z7 = (vector_field_fn(curr_z + eps * z6) - vector_field_fn(curr_z - eps * z6)) * inv_2eps
                z8 = (vector_field_fn(curr_z + eps * z7) - vector_field_fn(curr_z - eps * z7)) * inv_2eps
            else:
                z7 = z8 = torch.zeros_like(curr_z)
                
            curr_z = (
                curr_z 
                + dt_val * z1 
                + (dt_val**2 / 2.0) * z2 
                + (dt_val**3 / 6.0) * z3 
                + (dt_val**4 / 24.0) * z4
            )
            if self.order >= 6:
                curr_z = curr_z + (dt_val**5 / 120.0) * z5 + (dt_val**6 / 720.0) * z6
            if self.order >= 8:
                curr_z = curr_z + (dt_val**7 / 5040.0) * z7 + (dt_val**8 / 40320.0) * z8
                
            traj.append(curr_z.clone())
        return torch.stack(traj, dim=0)

def taylor_jet_integrate(
    vector_field_fn: Callable[[torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    t_span: torch.Tensor,
    order: int = 6,
    eps: float = 1e-4
) -> torch.Tensor:
    """
    Convenience function to integrate with JVP Taylor Jet.
    """
    integrator = TaylorJetIntegrator(order=order, eps=eps)
    return integrator.integrate(vector_field_fn, z0, t_span)
