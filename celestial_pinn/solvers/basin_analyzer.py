import numpy as np
from typing import Dict, Tuple

class BasinEntropyAnalyzer:
    """
    Computes Basin Entropy (S_b) and Boundary Uncertainty (Fractal Dimension).
    Reference: Daza, A., Wagemakers, A., Sanjuán, M. A. F. (Scientific Reports, 2016)
               Kumar, V., et al. (New Astronomy, 2021)
    """
    @staticmethod
    def compute_basin_entropy(basin_map: np.ndarray, box_size: int = 5) -> float:
        """
        Calculates the Basin Entropy S_b by partitioning the 2D grid into epsilon-boxes
        and computing Gibbs-Shannon entropy over attractor probability distributions.
        """
        Ny, Nx = basin_map.shape
        num_boxes_y = Ny // box_size
        num_boxes_x = Nx // box_size
        
        total_entropy = 0.0
        total_boxes = num_boxes_y * num_boxes_x
        
        if total_boxes == 0:
            return 0.0
            
        for by in range(num_boxes_y):
            for bx in range(num_boxes_x):
                sub_box = basin_map[by*box_size : (by+1)*box_size, bx*box_size : (bx+1)*box_size]
                valid_mask = sub_box >= 0
                if not np.any(valid_mask):
                    continue
                    
                vals, counts = np.unique(sub_box[valid_mask], return_counts=True)
                probs = counts / np.sum(counts)
                
                # Shannon entropy for this box: - sum p_i * log2(p_i)
                box_s = -np.sum(probs * np.log2(probs + 1e-12))
                total_entropy += box_s
                
        S_b = total_entropy / total_boxes
        return float(S_b)

    @staticmethod
    def compute_boundary_unpredictability(basin_map: np.ndarray) -> float:
        """
        Calculates the fraction of boundary points with mixed attractor neighbors.
        """
        Ny, Nx = basin_map.shape
        # Compute gradient / differences
        diff_x = np.abs(np.diff(basin_map, axis=1))
        diff_y = np.abs(np.diff(basin_map, axis=0))
        
        boundary_x = np.sum(diff_x > 0)
        boundary_y = np.sum(diff_y > 0)
        total_points = Ny * Nx
        
        return float((boundary_x + boundary_y) / (2.0 * total_points))
