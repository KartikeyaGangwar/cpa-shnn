import copy
import gc
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from ..models.as_pinn import AdaptiveSubspacePINN, SubspaceMLP
from ..models.conflict_monitor import ContinuousConflictMonitor
from ..physics.base_celestial import BaseCelestialSystem

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class TwoStageCelestialASPINNTrainer:
    def __init__(
        self,
        system: BaseCelestialSystem,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ):
        self.system = system
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        set_seed(self.seed)
        cleanup_gpu()

    def _compute_bandwidth(self) -> torch.Tensor:
        bounds = self.system.bounds
        # Calibrated bandwidth: 0.12 * domain width
        sigmas = [float(0.12 * (b[1] - b[0])) for b in bounds]
        return torch.tensor(sigmas, dtype=torch.float32, device=self.device).reshape(1, self.system.in_dim)

    def run_stage1_discovery(
        self,
        min_epochs: int = 200,
        max_epochs: int = 600,
        profile_freq: int = 15,
        warmup_epochs: int = 30,
        cooldown_epochs: int = 15,
        max_subspaces: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ) -> Dict:
        set_seed(self.seed)
        cleanup_gpu()
        sigma = self._compute_bandwidth()
        sigma_str = [round(float(s), 3) for s in sigma.cpu().numpy().ravel()]
        
        print(f"\n" + "="*70)
        print(f"  STAGE 1: AS-PINN AMR DISCOVERY (Seed: {self.seed})")
        print(f"  System: {self.system.name} | Device: {self.device} | Bandwidth: {sigma_str} | Max N*: {max_subspaces}")
        print("="*70)
        
        bounds = self.system.bounds
        c0 = torch.tensor([[0.5 * (b[0] + b[1]) for b in bounds]], dtype=torch.float32, device=self.device)
        
        probe_model = AdaptiveSubspacePINN(
            in_dim=self.system.in_dim,
            out_dim=self.system.out_dim,
            initial_subspaces=1,
            hidden_dim=32,
            layers=2,
            activation="tanh",
            bandwidth=sigma,
            initial_centroids=c0,
            u0=self.system.u0,
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(probe_model.parameters(), lr=lr, weight_decay=weight_decay)
        monitor = ContinuousConflictMonitor(
            threshold=0.0,
            ema_decay=0.85,
            min_clash_ratio=0.15,
            in_dim=self.system.in_dim,
        )
        
        last_cleavage_epoch = -cooldown_epochs
        consecutive_stable_cycles = 0
        cleavage_history = []
        start_time = time.time()
        
        for epoch in range(1, max_epochs + 1):
            probe_model.train()
            optimizer.zero_grad()
            
            t_int = self.system.sample_interior(2048)
            res = self.system.compute_residuals(probe_model, t_int)
            loss_res = torch.mean(res ** 2)
            
            loss_energy = self.system.compute_energy_conservation_loss(probe_model, t_int)
            
            total_loss = loss_res + 5.0 * loss_energy
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(probe_model.parameters(), max_norm=5.0)
            optimizer.step()
            
            can_cleave = (
                epoch >= warmup_epochs and
                (epoch - last_cleavage_epoch) >= cooldown_epochs and
                probe_model.num_subspaces < max_subspaces
            )
            
            cleaved_this_epoch = False
            
            if can_cleave and (epoch % profile_freq == 0):
                t_prof = self.system.sample_interior(128)
                
                with torch.no_grad():
                    psi_prof = probe_model.partition_of_unity(t_prof)
                
                for k in range(probe_model.num_subspaces):
                    sub_model = probe_model.subspaces[k]
                    sub_params = [p for p in sub_model.parameters() if p.requires_grad]
                    if not sub_params:
                        continue
                        
                    c_k = probe_model.centroids[k:k+1]
                    sigma_k = probe_model.bandwidth
                    
                    active_mask = psi_prof[:, k] > 0.05
                    if probe_model.num_subspaces > 1 and torch.sum(active_mask) >= 6:
                        t_active = t_prof[active_mask][:48]
                    else:
                        t_active = t_prof[:48]
                        
                    per_sample_grads = []
                    valid_pts = []
                    
                    for i in range(t_active.shape[0]):
                        pt = t_active[i:i+1].clone().detach().requires_grad_(True)
                        def sub_f(t_in):
                            u_local = sub_model((t_in - c_k) / sigma_k)
                            t_factor = 1.0 - torch.exp(-torch.clamp(t_in, min=0.0))
                            return self.system.u0 + t_factor * u_local
                            
                        r = self.system.compute_residuals(sub_f, pt)
                        l_pt = torch.mean(r ** 2)
                        
                        grads = torch.autograd.grad(l_pt, sub_params, retain_graph=False, create_graph=False, allow_unused=True)
                        flat_g = torch.cat([
                            (g.view(-1) if g is not None else torch.zeros(p.numel(), device=self.device))
                            for g, p in zip(grads, sub_params)
                        ])
                        per_sample_grads.append(flat_g)
                        valid_pts.append(pt.squeeze(0))
                        
                    if len(per_sample_grads) > 4:
                        sample_grads_t = torch.stack(per_sample_grads, dim=0)
                        sample_pts_t = torch.stack(valid_pts, dim=0)
                        
                        C, mean_align, clash_ratio, pt_conflict = monitor.analyze_gram_matrix(sample_grads_t)
                        
                        if k not in monitor.ema_conflict_score:
                            monitor.ema_conflict_score[k] = mean_align
                            monitor.ema_clash_ratio[k] = clash_ratio
                        else:
                            monitor.ema_conflict_score[k] = (
                                monitor.ema_decay * monitor.ema_conflict_score[k] +
                                (1.0 - monitor.ema_decay) * mean_align
                            )
                            monitor.ema_clash_ratio[k] = (
                                monitor.ema_decay * monitor.ema_clash_ratio[k] +
                                (1.0 - monitor.ema_decay) * clash_ratio
                            )
                            
                        ema_align = monitor.ema_conflict_score[k]
                        ema_clash = monitor.ema_clash_ratio[k]
                        should_cleave = (ema_clash >= 0.12) or (ema_align < 0.20)
                        
                        if should_cleave and probe_model.num_subspaces < max_subspaces:
                            child_centroid = monitor.locate_clashing_center(sample_pts_t, pt_conflict).detach().to(self.device)
                            norm_dists = torch.norm((probe_model.centroids - child_centroid) / probe_model.bandwidth, dim=-1)
                            
                            if torch.min(norm_dists) < 0.30:
                                continue
                                
                            probe_model.spawn_new_subspace(
                                centroid=child_centroid,
                                bandwidth=sigma,
                                parent_idx=k,
                            )
                            optimizer = torch.optim.AdamW(probe_model.parameters(), lr=lr, weight_decay=weight_decay)
                            c_np = child_centroid.detach().cpu().numpy().round(3).tolist()
                            print(f"  [Discovery @ Epoch {epoch:4d}] Subspace {k} Clashing (Align: {ema_align:.2f}, Clash: {ema_clash:.2f}) -> Cleaved Subspace {probe_model.num_subspaces-1} @ {c_np}")
                            
                            cleavage_history.append({
                                "epoch": epoch,
                                "parent": k,
                                "child": probe_model.num_subspaces - 1,
                                "centroid": child_centroid.detach().cpu(),
                            })
                            last_cleavage_epoch = epoch
                            cleaved_this_epoch = True
                            consecutive_stable_cycles = 0
                            break
                            
            if (epoch >= min_epochs) and (epoch % profile_freq == 0) and not cleaved_this_epoch:
                consecutive_stable_cycles += 1
                if consecutive_stable_cycles >= 3:
                    print(f"  [+] Autonomous Physical Quiescence Achieved @ Epoch {epoch}!")
                    print(f"  [+] Final Discovered Subspaces: N* = {probe_model.num_subspaces}")
                    break
                    
        elapsed_stage1 = time.time() - start_time
        info = {
            "num_subspaces": probe_model.num_subspaces,
            "centroids": probe_model.centroids.clone().detach(),
            "bandwidth": sigma,
            "cleavage_history": cleavage_history,
            "stage1_time": elapsed_stage1,
        }
        del probe_model, optimizer, monitor
        cleanup_gpu()
        return info

    def train_stage2_production(
        self,
        discovered_info: Dict,
        hidden_dim: int = 48,
        layers: int = 3,
        adamw_epochs: int = 800,
        lbfgs_steps: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        n_interior: int = 4096,
        eval_freq: int = 50,
    ) -> Tuple[nn.Module, Dict]:
        set_seed(self.seed)
        cleanup_gpu()
        
        num_subspaces = discovered_info["num_subspaces"]
        centroids = discovered_info["centroids"]
        bandwidth = discovered_info["bandwidth"]
        bw_str = [round(float(b), 3) for b in bandwidth.cpu().numpy().ravel()]
        
        print("\n" + "="*65)
        print("  STAGE 2: AS-PINN PRODUCTION TRAINING (AdamW + Combined L-BFGS Polish)")
        print(f"  System: {self.system.name} | Subspaces N*={num_subspaces} | Bandwidth: {bw_str}")
        print("="*65)
        
        model = AdaptiveSubspacePINN(
            in_dim=self.system.in_dim,
            out_dim=self.system.out_dim,
            initial_subspaces=num_subspaces,
            hidden_dim=hidden_dim,
            layers=layers,
            activation="tanh",
            bandwidth=bandwidth,
            initial_centroids=centroids,
            u0=self.system.u0,
        ).to(self.device)
        
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [+] Clean N* Model Initialized with {n_params} parameters.")
        
        history = {
            "epoch": [],
            "loss_total": [],
            "loss_res": [],
            "loss_energy": [],
            "rel_l2_err": [],
            "wall_time": [],
        }
        
        start_time = time.time()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=adamw_epochs, eta_min=1e-5)
        
        best_total_loss = float("inf")
        best_state = None
        
        print(f"\n  [Phase A] Running Synchronized AdamW Optimization ({adamw_epochs} epochs)...")
        for epoch in range(1, adamw_epochs + 1):
            model.train()
            optimizer.zero_grad()
            
            t_int = self.system.sample_interior(n_interior)
            res = self.system.compute_residuals(model, t_int)
            loss_res = torch.mean(res ** 2)
            
            loss_energy = self.system.compute_energy_conservation_loss(model, t_int)
            
            total_loss = loss_res + 5.0 * loss_energy
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            
            if epoch % eval_freq == 0 or epoch == adamw_epochs:
                elapsed = time.time() - start_time
                tot_val = total_loss.item()
                rel_err = self.system.compute_relative_l2_error(model, n_test=1000)
                
                history["epoch"].append(epoch)
                history["loss_total"].append(tot_val)
                history["loss_res"].append(loss_res.item())
                history["loss_energy"].append(loss_energy.item())
                history["rel_l2_err"].append(rel_err)
                history["wall_time"].append(elapsed)
                
                if tot_val < best_total_loss:
                    best_total_loss = tot_val
                    best_state = copy.deepcopy(model.state_dict())
                    
                print(f"  [AdamW Epoch {epoch:4d}/{adamw_epochs}] Total Loss: {tot_val:.6e} | Res: {loss_res.item():.6e} | Rel L2: {rel_err*100:.3f}% | Time: {elapsed:.1f}s")
                
        # Phase B: Combined Global L-BFGS Polish
        if lbfgs_steps > 0:
            print(f"\n  [Phase B] Running Full-Horizon Combined Global L-BFGS Polish ({lbfgs_steps} steps)...")
            lbfgs_opt = torch.optim.LBFGS(
                model.parameters(),
                lr=1.0,
                max_iter=lbfgs_steps,
                max_eval=int(lbfgs_steps * 1.5),
                history_size=50,
                tolerance_grad=1e-12,
                tolerance_change=1e-16,
                line_search_fn="strong_wolfe",
            )
            
            t_int_full = self.system.sample_interior(n_interior)
            lbfgs_step = [0]
            
            def closure():
                lbfgs_opt.zero_grad()
                res = self.system.compute_residuals(model, t_int_full)
                l_res = torch.mean(res ** 2)
                l_energy = self.system.compute_energy_conservation_loss(model, t_int_full)
                tot = l_res + 5.0 * l_energy
                tot.backward()
                
                lbfgs_step[0] += 1
                tot_val = tot.item()
                
                if lbfgs_step[0] % 10 == 0 or lbfgs_step[0] == 1:
                    rel_err = self.system.compute_relative_l2_error(model, n_test=1000)
                    history["epoch"].append(adamw_epochs + lbfgs_step[0])
                    history["loss_total"].append(tot_val)
                    history["loss_res"].append(l_res.item())
                    history["loss_energy"].append(l_energy.item())
                    history["rel_l2_err"].append(rel_err)
                    history["wall_time"].append(time.time() - start_time)
                    print(f"    [Combined L-BFGS Step {lbfgs_step[0]:3d}] Total Loss: {tot_val:.6e} | Res: {l_res.item():.6e} | Rel L2: {rel_err*100:.3f}%")
                return tot
                
            lbfgs_opt.step(closure)
            
        final_rel_err = self.system.compute_relative_l2_error(model, n_test=2000)
        final_time = history["wall_time"][-1]
        final_loss = history["loss_total"][-1]
        
        print(f"  [+] Production Complete! Final Loss: {final_loss:.6e} | Rel L2 Error: {final_rel_err*100:.4f}% | Total Time: {final_time:.1f}s")
        
        summary = {
            "final_loss_total": final_loss,
            "final_rel_l2_err": final_rel_err,
            "total_params": n_params,
            "wall_time": final_time,
            "history": history,
            "num_subspaces": num_subspaces,
        }
        cleanup_gpu()
        return model, summary
