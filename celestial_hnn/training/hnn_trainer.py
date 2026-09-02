import time
import copy
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from ..models.hnn import HamiltonianNeuralNetwork
from ..physics.base_hamiltonian import BaseHamiltonianSystem

class BaselineVectorFieldMLP(nn.Module):
    """Non-Hamiltonian Standard Vector Field Network (Predicts dz/dt directly without Symplectic structure)."""
    def __init__(self, state_dim: int, hidden_dim: int = 128, layers: int = 4):
        super().__init__()
        self.state_dim = state_dim
        net = [nn.Linear(state_dim, hidden_dim), nn.Tanh()]
        for _ in range(layers - 2):
            net.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        net.append(nn.Linear(hidden_dim, state_dim))
        self.net = nn.Sequential(*net)

    def time_derivative(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

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


class CelestialHNNTrainer:
    """
    Symplectic Hamiltonian Neural Network Benchmark Trainer with Trajectory Rollout.
    """
    def __init__(self, system: BaseHamiltonianSystem, device: Optional[torch.device] = None, seed: int = 42):
        self.system = system
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        torch.manual_seed(seed)
        
        # Pre-sample dense reference orbit data for high-precision trajectory anchoring
        t_dense = torch.linspace(0, self.system.T_max, 2000, device=self.device)
        self.orbit_z = self.system.ground_truth_trajectory(t_dense) # (2000, 2d)
        self.orbit_dz = self.system.canonical_derivatives(self.orbit_z) # (2000, 2d)

    def train_baseline_mlp(
        self,
        hidden_dim: int = 128,
        layers: int = 4,
        epochs: int = 1500,
        lr: float = 2e-3,
        n_samples: int = 4096,
    ) -> Tuple[nn.Module, Dict]:
        print(f"\n--- Training Standard Vector Field Baseline for {self.system.name} ---")
        model = BaselineVectorFieldMLP(self.system.state_dim, hidden_dim, layers).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        
        start_time = time.time()
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            
            # Mix random domain samples with orbit samples
            z_rand, dz_rand = self.system.sample_phase_space(n_samples // 2)
            idx_orbit = torch.randint(0, len(self.orbit_z), (n_samples // 2,), device=self.device)
            z_orb = self.orbit_z[idx_orbit]
            dz_orb = self.orbit_dz[idx_orbit]
            
            z_batch = torch.cat([z_rand, z_orb], dim=0)
            dz_true = torch.cat([dz_rand, dz_orb], dim=0)
            
            dz_pred = model.time_derivative(z_batch)
            loss = torch.mean((dz_pred - dz_true) ** 2)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if epoch % 300 == 0 or epoch == epochs:
                print(f"  [Standard MLP Epoch {epoch:4d}/{epochs}] Field Loss: {loss.item():.6e}")
                
        rel_l2 = self.system.compute_trajectory_error(model)
        elapsed = time.time() - start_time
        print(f"  [+] Standard Baseline Complete! Trajectory Rel L2: {rel_l2*100:.3f}% | Time: {elapsed:.1f}s")
        return model, {"final_loss": loss.item(), "rel_l2_error": rel_l2, "time": elapsed}

    def train_hnn(
        self,
        hidden_dim: int = 128,
        layers: int = 4,
        epochs: int = 1500,
        lr: float = 2e-3,
        n_samples: int = 4096,
    ) -> Tuple[nn.Module, Dict]:
        print(f"\n--- Training Symplectic Hamiltonian Neural Network (HNN) for {self.system.name} ---")
        model = HamiltonianNeuralNetwork(
            spatial_dim=self.system.spatial_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            activation="tanh",
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        
        start_time = time.time()
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            
            # Mix domain samples with orbit samples
            z_rand, dz_rand = self.system.sample_phase_space(n_samples // 2)
            idx_orbit = torch.randint(0, len(self.orbit_z), (n_samples // 2,), device=self.device)
            z_orb = self.orbit_z[idx_orbit]
            dz_orb = self.orbit_dz[idx_orbit]
            
            z_batch = torch.cat([z_rand, z_orb], dim=0)
            dz_true = torch.cat([dz_rand, dz_orb], dim=0)
            
            dz_pred = model.time_derivative(z_batch, create_graph=True)
            loss_field = torch.mean((dz_pred - dz_true) ** 2)
            
            # Energy conservation regularization: H(z(t)) must be close to exact H(z(t))
            H_pred = model.hamiltonian(z_orb)
            H_true = self.system.exact_hamiltonian(z_orb)
            loss_energy = torch.mean((H_pred - H_true) ** 2)
            
            loss = loss_field + 0.10 * loss_energy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            
            if epoch % 300 == 0 or epoch == epochs:
                print(f"  [HNN Symplectic Epoch {epoch:4d}/{epochs}] Symplectic Field Loss: {loss_field.item():.6e} | Energy Loss: {loss_energy.item():.6e}")
                
        rel_l2 = self.system.compute_trajectory_error(model)
        elapsed = time.time() - start_time
        print(f"  [+] HNN Symplectic Complete! Trajectory Rel L2: {rel_l2*100:.4f}% | Time: {elapsed:.1f}s")
        return model, {"final_loss": loss.item(), "rel_l2_error": rel_l2, "time": elapsed}
