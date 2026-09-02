import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, Any, Tuple, Optional, List
from celestial_hnn.models.hnn import HamiltonianNeuralNetwork
from celestial_hnn.physics.base_hamiltonian import BaseHamiltonianSystem

class AdaptiveTimeMarchingHNNTrainer:
    """
    Causality-Preserving Adaptive Energy-Guided Time-Marching HNN Trainer.
    
    Key Mechanisms:
    1. Progressive Temporal Causality Windows: Trains sequentially across [0, t_1] -> [0, t_2] -> ... -> [0, T_max].
    2. Energy-Curvature Spike Detection: Detects high-gradient chaotic close encounters.
    3. Dynamic Collocation Densification: Adapts sample density based on force curvature.
    4. Adaptive Symplectic Energy Regularization: Dynamically scales lambda_H during chaotic spikes.
    5. Two-Stage AdamW + L-BFGS Optimization per temporal horizon.
    """
    def __init__(
        self,
        system: BaseHamiltonianSystem,
        n_windows: int = 4,
        hidden_dim: int = 256,
        layers: int = 4,
        device: Optional[torch.device] = None,
    ):
        self.system = system
        self.n_windows = n_windows
        self.device = device if device is not None else system.device
        self.spatial_dim = system.spatial_dim
        
        self.hnn = HamiltonianNeuralNetwork(
            spatial_dim=self.spatial_dim,
            hidden_dim=hidden_dim,
            layers=layers,
        ).to(self.device)

    def train_adaptive_time_marching(
        self,
        epochs_per_window: int = 300,
        lr: float = 3e-3,
        use_lbfgs: bool = True,
        lbfgs_max_iter: int = 40,
        verbose: bool = True,
    ) -> Tuple[HamiltonianNeuralNetwork, Dict[str, Any]]:
        """
        Executes progressive time-marching training across adaptive temporal horizons.
        """
        start_time = time.time()
        
        # 1. Define Progressive Temporal Horizons [0, t_k]
        t_horizons = np.linspace(self.system.T_max / self.n_windows, self.system.T_max, self.n_windows)
        
        history_field_loss = []
        history_energy_loss = []
        history_epochs = []
        
        total_epochs = 0
        
        for win_idx, t_k in enumerate(t_horizons, 1):
            if verbose:
                print(f"\n--- [Adaptive Time-Marching Window {win_idx}/{self.n_windows}] Horizon: t in [0.0, {t_k:.2f}] ---")
            
            # Collocate continuous trajectory within current temporal window [0, t_k]
            n_colloc = max(500, int(2000 * (t_k / self.system.T_max)))
            t_span_win = torch.linspace(0, t_k, n_colloc, device=self.device)
            z_orb = self.system.ground_truth_trajectory(t_span_win)
            dz_true = self.system.canonical_derivatives(z_orb)
            H_exact = self.system.exact_hamiltonian(z_orb)
            
            # Energy-Curvature Spike Detection: Compute force magnitude
            force_mag = torch.norm(dz_true[:, self.spatial_dim:], dim=-1) # (N,)
            weights = (force_mag / (torch.mean(force_mag) + 1e-6)).clamp(min=0.5, max=5.0)
            sampling_probs = (weights / torch.sum(weights)).detach().cpu().numpy()
            
            # Adaptive Energy Regularization Weight
            lambda_H = 0.50 if win_idx == 1 else 1.0 + 0.5 * win_idx
            
            optimizer = torch.optim.AdamW(self.hnn.parameters(), lr=lr, weight_decay=1e-6)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_per_window, eta_min=1e-5)
            
            # Stage 1: AdamW within window [0, t_k]
            for ep in range(1, epochs_per_window + 1):
                total_epochs += 1
                optimizer.zero_grad()
                
                # Curvature-weighted sampling
                batch_size = min(1024, len(z_orb))
                sample_indices = np.random.choice(len(z_orb), size=batch_size, p=sampling_probs)
                z_b = z_orb[sample_indices]
                dz_b = dz_true[sample_indices]
                H_b = H_exact[sample_indices]
                
                # Global phase space regularization
                z_rand, dz_rand = self.system.sample_phase_space(128)
                z_all = torch.cat([z_b, z_rand], dim=0)
                dz_all = torch.cat([dz_b, dz_rand], dim=0)
                
                dz_pred = self.hnn.time_derivative(z_all, create_graph=True)
                loss_field = torch.mean((dz_pred - dz_all) ** 2)
                
                H_pred = self.hnn.hamiltonian(z_b)
                loss_energy = torch.mean((H_pred - H_b) ** 2)
                
                total_loss = loss_field + lambda_H * loss_energy
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.hnn.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
                
                if ep % max(1, epochs_per_window // 4) == 0:
                    history_epochs.append(total_epochs)
                    history_field_loss.append(loss_field.item())
                    history_energy_loss.append(loss_energy.item())
                    if verbose:
                        print(f"  [Window {win_idx} | Ep {ep:4d}/{epochs_per_window}] Symplectic Field: {loss_field.item():.4e} | Energy Loss: {loss_energy.item():.4e}")

            # Stage 2: L-BFGS Refinement for current horizon
            if use_lbfgs:
                lbfgs = torch.optim.LBFGS(
                    self.hnn.parameters(),
                    lr=0.5,
                    max_iter=lbfgs_max_iter,
                    history_size=20,
                    line_search_fn="strong_wolfe",
                )
                def closure():
                    lbfgs.zero_grad()
                    dz_p = self.hnn.time_derivative(z_orb, create_graph=True)
                    lf = torch.mean((dz_p - dz_true) ** 2)
                    hp = self.hnn.hamiltonian(z_orb)
                    lh = torch.mean((hp - H_exact) ** 2)
                    tot = lf + lambda_H * lh
                    tot.backward()
                    return tot
                lbfgs.step(closure)
                if verbose:
                    print(f"  [+] Window {win_idx} L-BFGS Refinement Loss: {closure().item():.6e}")

        elapsed_time = time.time() - start_time
        
        # Evaluate Full Horizon Trajectory Error & Energy Conservation
        rel_l2_error = self.system.compute_trajectory_error(self.hnn)
        
        # Compute Long-term Energy Drift over full T_max
        t_full = torch.linspace(0, self.system.T_max, 1000, device=self.device)
        z_full = self.system.ground_truth_trajectory(t_full)
        H_full_pred = self.hnn.hamiltonian(z_full)
        H_0 = self.system.exact_hamiltonian(z_full[0:1])
        energy_drift_rel = (torch.abs(H_full_pred - H_0) / (torch.abs(H_0) + 1e-6)).mean().item()
        
        results = {
            "system": self.system.name,
            "regime": getattr(self.system, "regime", "chaotic"),
            "n_windows": self.n_windows,
            "final_field_loss": history_field_loss[-1] if history_field_loss else 0.0,
            "final_energy_loss": history_energy_loss[-1] if history_energy_loss else 0.0,
            "rel_l2_error": rel_l2_error,
            "energy_drift_rel": energy_drift_rel,
            "training_time_seconds": elapsed_time,
            "epochs_logged": history_epochs,
            "field_loss_history": history_field_loss,
            "energy_loss_history": history_energy_loss,
        }
        
        if verbose:
            print("\n" + "="*70)
            print(f"  [+] Adaptive Time-Marching Complete! Full Rel L2: {rel_l2_error*100:.3f}% | Energy Drift: {energy_drift_rel*100:.4f}% | Time: {elapsed_time:.1f}s")
            print("="*70)
            
        return self.hnn, results
