import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Dict, Union


class SubspaceMLP(nn.Module):
    """
    Subspace Multi-Layer Perceptron (MLP) for localized coordinate representation.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 48,
        layers: int = 3,
        activation: str = "tanh",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        if activation.lower() == "tanh":
            act_cls = nn.Tanh
        elif activation.lower() == "gelu":
            act_cls = nn.GELU
        elif activation.lower() == "silu":
            act_cls = nn.SiLU
        else:
            act_cls = nn.Tanh
            
        net = [nn.Linear(in_dim, hidden_dim), act_cls()]
        for _ in range(layers - 2):
            net.extend([nn.Linear(hidden_dim, hidden_dim), act_cls()])
        net.append(nn.Linear(hidden_dim, out_dim))
        
        self.network = nn.Sequential(*net)
        self._init_weights()

    def _init_weights(self):
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class AdaptiveSubspacePINN(nn.Module):
    """
    Adaptive N-Subspace Physics-Informed Neural Network (AS-PINN) with Exact Hard IC Ansatz.
    
    Ansatz:
      u(t) = u_0 + (1 - exp(-t)) * sum_{k=1}^N psi_k(t) * Phi_k((t - c_k)/sigma)
    Guarantees u(0) == u_0 exactly with ZERO initial condition deviation!
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        initial_subspaces: int = 1,
        hidden_dim: int = 48,
        layers: int = 3,
        activation: str = "tanh",
        bandwidth: Union[float, List[float], torch.Tensor] = 0.50,
        initial_centroids: Optional[torch.Tensor] = None,
        u0: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.activation = activation
        
        if u0 is not None:
            self.register_buffer("u0", u0.clone().detach().to(dtype=torch.float32).reshape(1, out_dim))
        else:
            self.u0 = None
        
        if isinstance(bandwidth, (int, float)):
            bw_tensor = torch.full((1, in_dim), float(bandwidth), dtype=torch.float32)
        elif isinstance(bandwidth, (list, tuple)):
            bw_tensor = torch.tensor(bandwidth, dtype=torch.float32).reshape(1, in_dim)
        elif isinstance(bandwidth, torch.Tensor):
            bw_tensor = bandwidth.clone().detach().to(dtype=torch.float32).reshape(1, in_dim)
        else:
            bw_tensor = torch.full((1, in_dim), 0.50, dtype=torch.float32)
        self.register_buffer("bandwidth", bw_tensor)
        
        if initial_centroids is not None:
            centroids_t = initial_centroids.clone().detach().to(dtype=torch.float32).reshape(-1, in_dim)
            N = centroids_t.shape[0]
        else:
            centroids_t = torch.zeros((initial_subspaces, in_dim), dtype=torch.float32)
            N = initial_subspaces
            
        self.register_buffer("centroids", centroids_t)
        
        self.subspaces = nn.ModuleList([
            SubspaceMLP(in_dim, out_dim, hidden_dim, layers, activation)
            for _ in range(N)
        ])

    @property
    def num_subspaces(self) -> int:
        return len(self.subspaces)

    def partition_of_unity(self, x: torch.Tensor) -> torch.Tensor:
        diff = (x.unsqueeze(1) - self.centroids.unsqueeze(0)) / self.bandwidth.unsqueeze(0)
        dist_sq = torch.sum(diff ** 2, dim=-1)
        logits = -0.5 * dist_sq
        return torch.softmax(logits, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        psi = self.partition_of_unity(x)
        u_nn = torch.zeros(B, self.out_dim, device=x.device, dtype=x.dtype)
        
        for k in range(self.num_subspaces):
            x_local = (x - self.centroids[k:k+1]) / self.bandwidth
            u_k = self.subspaces[k](x_local)
            u_nn = u_nn + psi[:, k:k+1] * u_k
            
        if self.u0 is not None:
            # Hard IC Ansatz: (1 - exp(-t)) * u_nn + u0
            t_factor = 1.0 - torch.exp(-torch.clamp(x, min=0.0))
            return self.u0 + t_factor * u_nn
            
        return u_nn

    def forward_subspace(self, subspace_idx: int, x: torch.Tensor) -> torch.Tensor:
        x_local = (x - self.centroids[subspace_idx:subspace_idx+1]) / self.bandwidth
        return self.subspaces[subspace_idx](x_local)

    def spawn_new_subspace(
        self,
        centroid: torch.Tensor,
        bandwidth: Optional[Union[float, List[float], torch.Tensor]] = None,
        parent_idx: Optional[int] = None,
    ) -> int:
        device = self.centroids.device
        dtype = self.centroids.dtype
        centroid = centroid.to(device=device, dtype=dtype).reshape(1, self.in_dim)
        
        if bandwidth is not None:
            if isinstance(bandwidth, (int, float)):
                bw_tensor = torch.full((1, self.in_dim), float(bandwidth), device=device, dtype=dtype)
            elif isinstance(bandwidth, (list, tuple)):
                bw_tensor = torch.tensor(bandwidth, device=device, dtype=dtype).reshape(1, self.in_dim)
            else:
                bw_tensor = bandwidth.clone().detach().to(device=device, dtype=dtype).reshape(1, self.in_dim)
            self.register_buffer("bandwidth", bw_tensor)
            
        new_mlp = SubspaceMLP(
            self.in_dim,
            self.out_dim,
            self.hidden_dim,
            self.layers,
            self.activation,
        ).to(device=device, dtype=dtype)
        
        if parent_idx is not None and 0 <= parent_idx < len(self.subspaces):
            new_mlp.load_state_dict(self.subspaces[parent_idx].state_dict())
            
        self.subspaces.append(new_mlp)
        
        updated_centroids = torch.cat([self.centroids, centroid], dim=0)
        self.register_buffer("centroids", updated_centroids)
        
        return len(self.subspaces) - 1
