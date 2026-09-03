import torch
import numpy as np
from typing import Tuple, Optional
from scipy.integrate import solve_ivp

class VariableMassMagneticBinarySystem:
    """
    Non-Autonomous Celestial Problem:
    Photogravitational Magnetic Binary Problem with Variable Mass
    Reference:
      Vinay Kumar and Sawan Kumar Marig,
      "Effect of variable mass on N-R basins of convergence in photogravitational magnetic binary problem",
      Astronomy Reports, Vol. 67, No. 2, pp. 194-208, 2023.
      (Also: Kinematics & Physics of Celestial Bodies 39(6), 325-341, 2023).
      
    Physics:
      - Two primary stars with Jeans-Meshchersky decaying mass: m_1(t) = (1-mu)*e^(-alpha*t), m_2(t) = mu*e^(-alpha*t)
      - Photogravitational radiation pressure: q1, q2 in (0, 1]
      - Magnetic dipole moments: lambda_1, lambda_2 decaying with mass loss
      - Rotating synodic coordinate frame with angular frequency n.
      - Extended Phase Space: Z_ext = (x, y, t, px, py, pt) in R^6.
    """
    def __init__(
        self,
        mu: float = 0.2,
        alpha_decay: float = 0.05,
        q1: float = 0.90,
        q2: float = 0.85,
        lambda1: float = 0.02,
        lambda2: float = 0.01,
        regime: str = "chaotic",
        device: Optional[torch.device] = None
    ):
        self.name = "VariableMassMagneticBinary_NonAutonomous"
        self.mu = mu
        self.alpha = alpha_decay
        self.q1 = q1
        self.q2 = q2
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.regime = regime
        
        self.spatial_dim = 2
        self.ext_spatial_dim = 3 # (x, y, t)
        self.state_dim = 6 # (x, y, t, px, py, pt)
        self.device = device if device is not None else torch.device("cpu")
        
        self.n = 1.0 # Angular velocity of synodic frame
        self.T_max = 12.0 # Time horizon
        
        # Primary positions in rotating frame
        self.x1 = -self.mu
        self.x2 = 1.0 - self.mu
        
        if regime == "chaotic":
            self.x0_val = 0.45
            self.y0_val = 0.55
            self.px0_val = 0.25
            self.py0_val = -0.30
        else:
            self.x0_val = 0.50
            self.y0_val = 0.00
            self.px0_val = 0.00
            self.py0_val = 0.40
            
        # Initial potential and conjugate energy pt
        r1_0 = np.sqrt((self.x0_val - self.x1)**2 + self.y0_val**2 + 0.04)
        r2_0 = np.sqrt((self.x0_val - self.x2)**2 + self.y0_val**2 + 0.04)
        V0 = -(self.q1 * (1.0 - self.mu) / r1_0 + self.q2 * self.mu / r2_0 + self.lambda1 / (r1_0**3) + self.lambda2 / (r2_0**3))
        coriolis0 = self.n * (self.px0_val * self.y0_val - self.py0_val * self.x0_val)
        kinetic0 = 0.5 * (self.px0_val**2 + self.py0_val**2)
        H0 = kinetic0 + coriolis0 + V0
        self.pt0_val = -H0 # Initial contact energy invariant K = H + pt = 0
        
        self.z0 = torch.tensor([self.x0_val, self.y0_val, 0.0, self.px0_val, self.py0_val, self.pt0_val], dtype=torch.float32, device=self.device)

    def exact_potential(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r1 = torch.sqrt((x - self.x1)**2 + y**2 + 0.04)
        r2 = torch.sqrt((x - self.x2)**2 + y**2 + 0.04)
        decay = torch.exp(-self.alpha * t)
        v_base = (self.q1 * (1.0 - self.mu) / r1 + self.q2 * self.mu / r2 + self.lambda1 / (r1**3) + self.lambda2 / (r2**3))
        return -decay * v_base

    def exact_extended_hamiltonian(self, z_ext: torch.Tensor) -> torch.Tensor:
        x = z_ext[:, 0:1]
        y = z_ext[:, 1:2]
        t = z_ext[:, 2:3]
        px = z_ext[:, 3:4]
        py = z_ext[:, 4:5]
        pt = z_ext[:, 5:6]
        
        kinetic = 0.5 * (px**2 + py**2)
        coriolis = self.n * (px * y - py * x)
        V = self.exact_potential(x, y, t)
        return kinetic + coriolis + V + pt

    def canonical_derivatives(self, z_ext: torch.Tensor) -> torch.Tensor:
        x = z_ext[:, 0:1]
        y = z_ext[:, 1:2]
        t = z_ext[:, 2:3]
        px = z_ext[:, 3:4]
        py = z_ext[:, 4:5]
        pt = z_ext[:, 5:6]
        
        # 1. dx/dt, dy/dt (Theorem 1 Exact Coriolis-Kinetic Velocity)
        dx_dt = px + self.n * y
        dy_dt = py - self.n * x
        
        # 2. dt/dt (Theorem 2 Exact Unit Clock)
        dt_dt = torch.ones_like(t)
        
        # 3. Spatial Gradients of V(x, y, t)
        r1 = torch.sqrt((x - self.x1)**2 + y**2 + 0.04)
        r2 = torch.sqrt((x - self.x2)**2 + y**2 + 0.04)
        decay = torch.exp(-self.alpha * t)
        
        dv_dr1 = -(self.q1 * (1.0 - self.mu) / (r1**2) + 3.0 * self.lambda1 / (r1**4))
        dv_dr2 = -(self.q2 * self.mu / (r2**2) + 3.0 * self.lambda2 / (r2**4))
        
        dV_dx = -decay * (dv_dr1 * (x - self.x1) / r1 + dv_dr2 * (x - self.x2) / r2)
        dV_dy = -decay * (dv_dr1 * y / r1 + dv_dr2 * y / r2)
        
        # dpx/dt = n*py - ∂V/∂x
        dpx_dt = self.n * py - dV_dx
        # dpy/dt = -n*px - ∂V/∂y
        dpy_dt = -self.n * px - dV_dy
        
        # 4. Power exchange rate dpt/dt = -∂V/∂t = -alpha * V(x, y, t)
        V_val = self.exact_potential(x, y, t)
        dpt_dt = -self.alpha * V_val
        
        return torch.cat([dx_dt, dy_dt, dt_dt, dpx_dt, dpy_dt, dpt_dt], dim=-1)

    def ground_truth_trajectory(self, t_eval: torch.Tensor) -> torch.Tensor:
        t_np = t_eval.detach().cpu().numpy()
        
        def ode_func(t, state):
            x, y, px, py, pt = state
            r1 = np.sqrt((x - self.x1)**2 + y**2 + 0.04)
            r2 = np.sqrt((x - self.x2)**2 + y**2 + 0.04)
            decay = np.exp(-self.alpha * t)
            
            dv_dr1 = -(self.q1 * (1.0 - self.mu) / (r1**2) + 3.0 * self.lambda1 / (r1**4))
            dv_dr2 = -(self.q2 * self.mu / (r2**2) + 3.0 * self.lambda2 / (r2**4))
            
            dV_dx = -decay * (dv_dr1 * (x - self.x1) / r1 + dv_dr2 * (x - self.x2) / r2)
            dV_dy = -decay * (dv_dr1 * y / r1 + dv_dr2 * y / r2)
            
            dx = px + self.n * y
            dy = py - self.n * x
            dpx = self.n * py - dV_dx
            dpy = -self.n * px - dV_dy
            
            v_base = (self.q1 * (1.0 - self.mu) / r1 + self.q2 * self.mu / r2 + self.lambda1 / (r1**3) + self.lambda2 / (r2**3))
            V_val = -decay * v_base
            dpt = -self.alpha * V_val
            return [dx, dy, dpx, dpy, dpt]
            
        y0 = [self.x0_val, self.y0_val, self.px0_val, self.py0_val, self.pt0_val]
        sol = solve_ivp(ode_func, [t_np[0], t_np[-1]], y0, t_eval=t_np, rtol=1e-11, atol=1e-13, method="DOP853")
        
        x_s, y_s, px_s, py_s, pt_s = sol.y
        t_s = t_np
        traj_np = np.column_stack([x_s, y_s, t_s, px_s, py_s, pt_s])
        return torch.tensor(traj_np, dtype=torch.float32, device=self.device)

    def compute_trajectory_error(self, model) -> float:
        t_dense = torch.linspace(0, self.T_max, 2500, device=self.device)
        z_gt = self.ground_truth_trajectory(t_dense)
        z_pred = model.integrate_symplectic_rk4(self.z0, t_dense).squeeze(1)
        num = torch.norm(z_pred[:, :2] - z_gt[:, :2])
        den = torch.norm(z_gt[:, :2]) + 1e-7
        return (num / den).item()
