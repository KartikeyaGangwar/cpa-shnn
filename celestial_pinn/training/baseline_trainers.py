import copy
import time
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from ..physics.base_celestial import BaseCelestialSystem

class StandardCelestialPINN(nn.Module):
    """Standard global MLP baseline with Hard IC support."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64, layers: int = 4, u0: Optional[torch.Tensor] = None):
        super().__init__()
        net = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(layers - 2):
            net.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        net.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*net)
        if u0 is not None:
            self.register_buffer("u0", u0.clone().detach().to(dtype=torch.float32).reshape(1, out_dim))
        else:
            self.u0 = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u_nn = self.net(x)
        if self.u0 is not None:
            t_factor = 1.0 - torch.exp(-torch.clamp(x, min=0.0))
            return self.u0 + t_factor * u_nn
        return u_nn


class BaselineCelestialTrainer:
    def __init__(self, system: BaseCelestialSystem, device: Optional[torch.device] = None, seed: int = 42):
        self.system = system
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        torch.manual_seed(seed)
        
    def train_standard_pinn(
        self,
        hidden_dim: int = 64,
        layers: int = 4,
        epochs: int = 800,
        lr: float = 1e-3,
        n_interior: int = 4096,
    ) -> Tuple[nn.Module, Dict]:
        model = StandardCelestialPINN(
            self.system.in_dim,
            self.system.out_dim,
            hidden_dim,
            layers,
            u0=self.system.u0
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        
        start_time = time.time()
        history = {"epoch": [], "loss_total": [], "rel_l2_err": [], "wall_time": []}
        
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            
            t_int = self.system.sample_interior(n_interior)
            res = self.system.compute_residuals(model, t_int)
            loss_res = torch.mean(res ** 2)
            
            loss_energy = self.system.compute_energy_conservation_loss(model, t_int)
            
            loss = loss_res + 5.0 * loss_energy
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if epoch % 100 == 0 or epoch == epochs:
                elapsed = time.time() - start_time
                rel_err = self.system.compute_relative_l2_error(model, n_test=1000)
                history["epoch"].append(epoch)
                history["loss_total"].append(loss.item())
                history["rel_l2_err"].append(rel_err)
                history["wall_time"].append(elapsed)
                print(f"  [Standard PINN Epoch {epoch:4d}/{epochs}] Loss: {loss.item():.6e} | Rel L2: {rel_err*100:.3f}% | Time: {elapsed:.1f}s")
                
        final_rel_err = self.system.compute_relative_l2_error(model, n_test=2000)
        summary = {
            "final_loss_total": history["loss_total"][-1],
            "final_rel_l2_err": final_rel_err,
            "wall_time": time.time() - start_time,
            "history": history,
        }
        return model, summary
