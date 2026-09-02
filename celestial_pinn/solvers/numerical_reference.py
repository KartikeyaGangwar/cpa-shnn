import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Tuple, List, Dict, Optional

class HighPrecisionCelestialIntegrator:
    """
    High-precision adaptive Symplectic / Explicit Runge-Kutta DOP853 solver
    with absolute and relative tolerances clamped at 1e-12.
    """
    def __init__(self, rtol: float = 1e-12, atol: float = 1e-12):
        self.rtol = rtol
        self.atol = atol

    def integrate_trajectory(
        self,
        rhs_fn: Callable[[float, np.ndarray], np.ndarray],
        t_span: Tuple[float, float],
        y0: np.ndarray,
        t_eval: Optional[np.ndarray] = None,
        method: str = "DOP853",
    ):
        sol = solve_ivp(
            rhs_fn,
            t_span,
            y0,
            method=method,
            rtol=self.rtol,
            atol=self.atol,
            t_eval=t_eval,
            dense_output=True,
        )
        return sol


class MultivariateNewtonRaphsonBasinSolver:
    """
    High-precision multivariate Newton-Raphson solver for libration points and
    fractal basins of attraction in multi-body effective potentials.
    
    Iteration:
      x_{n+1} = x_n - [J(x_n)]^{-1} * grad_Omega(x_n)
      where J(x) = Hessian(Omega_p).
    """
    def __init__(
        self,
        grad_fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
        hessian_fn: Optional[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
        max_iter: int = 100,
        tol: float = 1e-8,
    ):
        self.grad_fn = grad_fn
        self.hessian_fn = hessian_fn
        self.max_iter = max_iter
        self.tol = tol

    def _numerical_hessian(self, x: np.ndarray, y: np.ndarray, eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gx_plus, gy_plus_x = self.grad_fn(x + eps, y)
        gx_minus, gy_minus_x = self.grad_fn(x - eps, y)
        gx_yplus, gy_plus = self.grad_fn(x, y + eps)
        gx_yminus, gy_minus = self.grad_fn(x, y - eps)
        
        hxx = (gx_plus - gx_minus) / (2.0 * eps)
        hyy = (gy_plus - gy_minus) / (2.0 * eps)
        hxy = (gx_yplus - gx_yminus) / (2.0 * eps)
        return hxx, hxy, hyy

    def solve_grid_basins(
        self,
        x_grid: np.ndarray, # 1D array of x coordinates
        y_grid: np.ndarray, # 1D array of y coordinates
        known_attractors: List[Tuple[float, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized 2D grid basin calculation.
        
        Returns:
            basin_map: [Ny, Nx] integer array of attractor indices (0 to K-1, or -1 for escape/non-converged).
            iter_map: [Ny, Nx] integer array of iterations required to converge.
        """
        Nx = len(x_grid)
        Ny = len(y_grid)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        X_cur = X.copy()
        Y_cur = Y.copy()
        
        basin_map = np.full((Ny, Nx), -1, dtype=np.int32)
        iter_map = np.full((Ny, Nx), self.max_iter, dtype=np.int32)
        active_mask = np.ones((Ny, Nx), dtype=bool)
        
        attractors_arr = np.array(known_attractors) # [K, 2]
        K = len(known_attractors)
        
        for it in range(1, self.max_iter + 1):
            if not np.any(active_mask):
                break
                
            x_act = X_cur[active_mask]
            y_act = Y_cur[active_mask]
            
            gx, gy = self.grad_fn(x_act, y_act)
            
            if self.hessian_fn is not None:
                hxx, hxy, hyy = self.hessian_fn(x_act, y_act)
            else:
                hxx, hxy, hyy = self._numerical_hessian(x_act, y_act)
                
            det = hxx * hyy - hxy ** 2
            det_safe = np.where(np.abs(det) < 1e-12, 1e-12, det)
            
            # [H]^-1 * [gx, gy]^T
            dx = (hyy * gx - hxy * gy) / det_safe
            dy = (-hxy * gx + hxx * gy) / det_safe
            
            X_cur[active_mask] -= dx
            Y_cur[active_mask] -= dy
            
            # Check convergence to known attractors
            x_upd = X_cur[active_mask]
            y_upd = Y_cur[active_mask]
            
            # Distance to each attractor
            # x_upd: [M], attractors: [K, 2]
            dists = np.sqrt((x_upd[:, None] - attractors_arr[:, 0])**2 + (y_upd[:, None] - attractors_arr[:, 1])**2)
            min_dists = np.min(dists, axis=1)
            closest_attr = np.argmin(dists, axis=1)
            
            converged = min_dists < self.tol
            
            # Update maps
            act_indices = np.where(active_mask)
            conv_act_idx = (act_indices[0][converged], act_indices[1][converged])
            
            basin_map[conv_act_idx] = closest_attr[converged]
            iter_map[conv_act_idx] = it
            
            # Deactivate converged points
            active_mask[conv_act_idx] = False
            
        return basin_map, iter_map
