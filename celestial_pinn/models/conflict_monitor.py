import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Tuple

class ContinuousConflictMonitor:
    """
    Continuous Intra-Subspace Gradient Conflict Monitor for AS-PINN.
    """
    def __init__(
        self,
        threshold: float = 0.05,
        ema_decay: float = 0.85,
        min_clash_ratio: float = 0.20,
        in_dim: int = 2,
    ):
        dim_factor = (in_dim / 2.0) ** 0.5 if in_dim >= 2 else 1.0
        self.threshold = threshold / dim_factor
        self.min_clash_ratio = max(0.15, min_clash_ratio / dim_factor)
        self.ema_decay = ema_decay
        self.in_dim = in_dim
        self.ema_conflict_score: Dict[int, float] = {}
        self.ema_clash_ratio: Dict[int, float] = {}
        self.history: Dict[int, List[Dict[str, float]]] = {}

    def compute_per_point_gradients_vmap(
        self,
        subspace: nn.Module,
        x_batch: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        params = dict(subspace.named_parameters())
        param_names = list(params.keys())
        
        def single_point_eval(param_dict, x_single):
            u_single = torch.func.functional_call(
                subspace, param_dict, (x_single.unsqueeze(0),)
            ).squeeze(0)
            return loss_fn(u_single, x_single)

        grad_single_fn = torch.func.grad(single_point_eval, argnums=0)
        vmap_grad_fn = torch.func.vmap(grad_single_fn, in_dims=(None, 0))
        grads_dict = vmap_grad_fn(params, x_batch)
        
        flat_list = [grads_dict[name].reshape(x_batch.shape[0], -1) for name in param_names]
        G = torch.cat(flat_list, dim=1)
        return G

    def compute_per_point_gradients_autograd_fallback(
        self,
        subspace: nn.Module,
        x_batch: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        subspace_params = [p for p in subspace.parameters() if p.requires_grad]
        M = x_batch.shape[0]
        per_point_grads = []
        
        for i in range(M):
            xi = x_batch[i:i+1].clone().detach().requires_grad_(True)
            ui = subspace(xi).squeeze(0)
            loss = loss_fn(ui, xi.squeeze(0))
            grads = torch.autograd.grad(loss, subspace_params, retain_graph=False, create_graph=False)
            flat_grad = torch.cat([g.reshape(-1) for g in grads])
            per_sample_grads.append(flat_grad)
            
        return torch.stack(per_sample_grads)

    def analyze_gram_matrix(
        self,
        G: torch.Tensor,
        eps: float = 1e-8
    ) -> Tuple[torch.Tensor, float, float, torch.Tensor]:
        M, P = G.shape
        if M < 2:
            C = torch.ones((M, M), device=G.device)
            return C, 1.0, 0.0, torch.zeros(M, device=G.device)

        gnorms = torch.norm(G, p=2, dim=1, keepdim=True).clamp_min(eps)
        G_norm = G / gnorms
        
        C = torch.mm(G_norm, G_norm.t()).clamp(-1.0, 1.0)
        mask = ~torch.eye(M, dtype=torch.bool, device=G.device)
        off_diags = C[mask]
        
        mean_alignment = off_diags.mean().item()
        clash_ratio = (off_diags < 0.0).float().mean().item()
        
        C_no_diag = C.masked_fill(~mask, 0.0)
        neg_C = torch.clamp(C_no_diag, max=0.0)
        point_conflict_scores = -neg_C.sum(dim=1)
        
        return C, mean_alignment, clash_ratio, point_conflict_scores

    def locate_clashing_center(
        self,
        x_batch: torch.Tensor,
        point_conflict_scores: torch.Tensor,
        top_k_fraction: float = 0.35,
    ) -> torch.Tensor:
        M = x_batch.shape[0]
        k = max(1, int(M * top_k_fraction))
        
        if point_conflict_scores.max() == 0:
            return x_batch.mean(dim=0)
            
        top_indices = torch.topk(point_conflict_scores, k=k).indices
        clashing_points = x_batch[top_indices]
        clashing_weights = point_conflict_scores[top_indices].clamp_min(1e-6)
        
        weighted_center = (clashing_points * clashing_weights.unsqueeze(-1)).sum(dim=0) / clashing_weights.sum()
        return weighted_center

    def reset_subspace(self, subspace_idx: int):
        self.ema_conflict_score[subspace_idx] = 0.5
        self.ema_clash_ratio[subspace_idx] = 0.0
