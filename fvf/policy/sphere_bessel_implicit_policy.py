""" 
sphere_bessel_implicit_policy.py - Debug Version

Added extensive debugging to find why drone isn't moving.
"""

import io
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

from fvf.utils import torch_utils
from fvf.policy.base_policy import BasePolicy

from EquiHarmony.eharmony import grid, plotting


class SphereBesselImplicitPolicy(BasePolicy):
    """
    Sphere Bessel implicit policy with debugging.
    """

    def __init__(
        self,
        obs_encoder: nn.Module,
        energy_head: nn.Module,
        obs_dim: int,
        action_dim: int,
        num_obs_steps: int,
        num_action_steps: int,
        num_neg_act_samples: int,
        pred_n_samples: int,
        optimize_negatives: bool = False,
        sample_actions: bool = False,
        temperature: float = 1.0,
        grad_pen: bool = False,
    ):
        super().__init__(obs_dim, action_dim, num_obs_steps, num_action_steps)
        self.num_neg_act_samples = num_neg_act_samples
        self.pred_n_samples = pred_n_samples
        self.optimize_negatives = optimize_negatives
        self.sample_actions = sample_actions
        self.temperature = temperature
        self.grad_pen = grad_pen

        self.obs_encoder = obs_encoder
        self.energy_head = energy_head
        self.apply(torch_utils.init_weights)
        
        # Debug counter
        self._debug_counter = 0

    def get_action(self, obs, device, use_break=False):
        """Get the action for the observation - WITH DEBUGGING."""
        self._debug_counter += 1
        debug = (self._debug_counter <= 3)  # Debug first 3 calls
        
        if debug:
            print(f"\n{'='*60}")
            print(f"[DEBUG get_action] Call #{self._debug_counter}")
            print(f"{'='*60}")
        
        nobs = self.normalizer.normalize(obs)
        B = list(obs.values())[0].shape[0]

        
        action_stats = self.get_action_stats()
        
        if debug:
            print(f"[DEBUG] action_stats: min={action_stats['min'].tolist()}, max={action_stats['max'].tolist()}")
        
        if hasattr(self.energy_head, 'sbh'):
            sbh = self.energy_head.sbh

            r_min = action_stats["min"][0].item()
            r_max = action_stats["max"][0].item()
            
            if debug:
                print(f"[DEBUG] r range: [{r_min:.4f}, {r_max:.4f}]")
            
            # r = torch.linspace(r_min, r_max, sbh.n_k, device=device, dtype=torch.float32)
            r = torch.linspace(r_min + 0.01, r_max, sbh.n_k, device=device, dtype=torch.float32)

            theta_grid, phi_grid = sbh.grid
            theta_np = theta_grid[:, 0]
            phi_np = phi_grid[0, :]
            
            theta = torch.from_numpy(theta_np).to(device).float()
            phi = torch.from_numpy(phi_np).to(device).float()
            
            if debug:
                print(f"[DEBUG] Grid sizes: r={len(r)}, theta={len(theta)}, phi={len(phi)}")
                print(f"[DEBUG] theta range: [{theta.min():.4f}, {theta.max():.4f}]")
                print(f"[DEBUG] phi range: [{phi.min():.4f}, {phi.max():.4f}]")

            R, THETA, PHI = torch.meshgrid(r, theta, phi, indexing='ij')
            
            actions_grid = torch.stack([
                R.flatten(), 
                THETA.flatten(), 
                PHI.flatten()
            ], dim=1)
            
            N = actions_grid.shape[0]
            
            if debug:
                print(f"[DEBUG] Total grid points N = {N}")
            
            actions_grid = actions_grid.unsqueeze(0).expand(B, -1, -1)
            
            # Get observation features
            obs_feat = self.obs_encoder(nobs)
            
            if debug:
                print(f"[DEBUG] obs_feat shape: {obs_feat.shape}")
                print(f"[DEBUG] obs_feat stats: mean={obs_feat.mean():.4f}, std={obs_feat.std():.4f}")
            
            # Get energy
            logits, coeffs = self.energy_head(obs_feat, actions_grid, return_coeffs=True)
            
            if debug:
                print(f"[DEBUG] coeffs shape: {coeffs.shape}")
                print(f"[DEBUG] coeffs stats: mean={coeffs.mean():.4f}, std={coeffs.std():.4f}, "
                      f"min={coeffs.min():.4f}, max={coeffs.max():.4f}")
                print(f"[DEBUG] logits shape: {logits.shape}")
                print(f"[DEBUG] logits stats: mean={logits.mean():.4f}, std={logits.std():.4f}, "
                      f"min={logits.min():.4f}, max={logits.max():.4f}")
                print(f"[DEBUG] logits range (max-min): {(logits.max() - logits.min()):.4f}")
            
            # Softmax
            action_probs = torch.softmax(logits / self.temperature, dim=-1)
            
            if debug:
                print(f"[DEBUG] temperature = {self.temperature}")
                print(f"[DEBUG] action_probs stats: mean={action_probs.mean():.6f}, "
                      f"std={action_probs.std():.6f}, max={action_probs.max():.6f}")
                
                # Check entropy (uniformity)
                entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum(dim=-1)
                max_entropy = np.log(N)
                print(f"[DEBUG] Entropy: {entropy.mean():.4f} / {max_entropy:.4f} (max)")
                print(f"[DEBUG] Entropy ratio: {(entropy.mean() / max_entropy):.4f} (1.0 = uniform)")
            
            # Select action
            if self.sample_actions:
                flat_indexes = torch.multinomial(action_probs, num_samples=1).squeeze(-1)
            else:
                flat_indexes = torch.argmax(action_probs, dim=-1)
            
            if debug:
                print(f"[DEBUG] Selected indices: {flat_indexes.tolist()}")
            
            # Get best action in spherical coords
            best_actions_spherical = actions_grid[torch.arange(B, device=device), flat_indexes]
            
            if debug:
                print(f"[DEBUG] best_actions_spherical (r, θ, φ): {best_actions_spherical[0].tolist()}")
            
            # Unnormalize radius
            r_normalized = best_actions_spherical[:, 0:1]
            
            # Create dummy tensor for unnormalization
            dummy = torch.zeros(B, 1, 3, device=device)
            dummy[:, 0, 0] = r_normalized.squeeze()
            unnorm = self.normalizer["action"].unnormalize(dummy)
            r_unnorm = unnorm[:, 0, 0]
            
            if debug:
                print(f"[DEBUG] r_normalized: {r_normalized[0].item():.4f}")
                print(f"[DEBUG] r_unnormalized: {r_unnorm[0].item():.4f}")
            
            theta_best = best_actions_spherical[:, 1]
            phi_best = best_actions_spherical[:, 2]
            
            if debug:
                print(f"[DEBUG] theta_best: {theta_best[0].item():.4f}")
                print(f"[DEBUG] phi_best: {phi_best[0].item():.4f}")
            
            # Convert to Cartesian
            x = r_unnorm * torch.sin(theta_best) * torch.cos(phi_best)
            y = r_unnorm * torch.sin(theta_best) * torch.sin(phi_best)
            z = r_unnorm * torch.cos(theta_best)
            
            if debug:
                print(f"[DEBUG] Cartesian (x, y, z): ({x[0].item():.4f}, {y[0].item():.4f}, {z[0].item():.4f})")
            
            actions = torch.stack([x, y, z], dim=1).unsqueeze(1)
            
            if debug:
                print(f"[DEBUG] Final action shape: {actions.shape}")
                print(f"[DEBUG] Final action: {actions[0, 0].tolist()}")
                
                # Check if action is always the same
                if self._debug_counter > 1:
                    if hasattr(self, '_last_action'):
                        diff = (actions - self._last_action).abs().max().item()
                        print(f"[DEBUG] Diff from last action: {diff:.6f}")
                        if diff < 1e-6:
                            print("[DEBUG] Action is IDENTICAL to last action!")
                
                self._last_action = actions.clone()
            
            # Reshape energy for visualization
            energy_grid = logits.view(B, sbh.n_k, sbh.num_theta, sbh.num_phi)
            
            return {
                "action": actions,
                "energy": energy_grid,
                "fourier_coeffs": coeffs.cpu(),
            }
        
        else:
            raise ValueError("energy_head does not have sbh attribute")

    def compute_loss(self, batch):
        """Compute loss - unchanged from original."""
        nobs = self.normalizer.normalize(batch["obs"])
        naction = self.normalizer["action"].normalize(batch["action"]).float()

        To = self.num_obs_steps
        Ta = self.num_action_steps
        B = naction.shape[0]

        start = To - 1
        end = start + Ta
        naction = naction[:, start:end]

        noisy_actions = naction

        action_stats = self.get_action_stats()
        action_dist = torch.distributions.Uniform(
            low=action_stats["min"], high=action_stats["max"]
        )

        negatives = action_dist.sample((B, self.num_neg_act_samples, Ta)).to(
            dtype=naction.dtype
        )

        targets = torch.cat([noisy_actions.unsqueeze(1), negatives], dim=1)
        N = targets.size(1)

        permutation = torch.rand(B, N).argsort(dim=1)
        targets = targets[torch.arange(B).unsqueeze(-1), permutation]
        ground_truth = (permutation == 0).nonzero()[:, 1].to(naction.device)
        one_hot = F.one_hot(
            ground_truth, num_classes=self.num_neg_act_samples + 1
        ).float()

        r = targets[:, :, 0, 0]
        theta = self.normalizer["action"].unnormalize(targets)[:, :, 0, 1]
        phi = self.normalizer["action"].unnormalize(targets)[:, :, 0, 2]
        sphere_act = torch.concatenate(
            [r.view(B, N, 1), theta.view(B, N, 1), phi.view(B, N, 1)],
            axis=2,
        )

        obs_feat = self.obs_encoder(nobs)
        energy = self.energy_head(obs_feat, sphere_act)

        probs = F.log_softmax(energy, dim=1)
        ebm_loss = F.kl_div(probs, one_hot, reduction="batchmean")
        loss = ebm_loss

        return loss, ebm_loss, torch.tensor(0.0), torch.tensor(0.0)

    def get_action_stats(self):
        action_stats = self.normalizer["action"].get_output_stats()
        repeated_stats = dict()
        for key, value in action_stats.items():
            n_repeats = self.action_dim // value.shape[0]
            repeated_stats[key] = value.repeat(n_repeats)
        return repeated_stats