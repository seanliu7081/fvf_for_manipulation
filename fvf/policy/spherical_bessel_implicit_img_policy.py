"""
spherical_bessel_implicit_policy.py
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


class SphericalBesselImgImplicitPolicy(BasePolicy):
    """
    Spherical Bessel implicit policy.
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
        print(f"obs_encoder.z_dim: {getattr(obs_encoder, 'z_dim', 'N/A')}")
        print(f"energy_head input dim: {energy_head.energy_mlp.mlp[0].in_features}")

    def get_action(self, obs, device, use_break=False):
        # """Get the action for the observation."""
        # print("=== Eval Debug ===")
        # print(f"raw keypoints shape: {obs['keypoints'].shape}")
        # print(f"raw keypoints[0]: {obs['keypoints'][0]}")
        nobs = self.normalizer.normalize(obs)
        # print(f"normalized keypoints shape: {nobs['keypoints'].shape}")
        # print(f"normalized keypoints[0]: {nobs['keypoints'][0]}")
        # print("=== End Eval Debug ===")
        B = list(obs.values())[0].shape[0]

        action_stats = self.get_action_stats()

        # Access the spherical bessel harmonics from energy head
        sbh = self.energy_head.sbh

        # Create radius grid
        r_min = action_stats["min"][0].item()
        r_max = action_stats["max"][0].item()
        r = torch.linspace(r_min + 0.01, r_max, sbh.n_k, device=device, dtype=torch.float32)

        # Get angular grid
        theta_grid, phi_grid = sbh.grid
        theta_np = theta_grid[:, 0]
        phi_np = phi_grid[0, :]
        
        theta = torch.from_numpy(theta_np).to(device).float()
        phi = torch.from_numpy(phi_np).to(device).float()

        if use_break:
            breakpoint()

        # Create full 3D meshgrid of actions
        R, THETA, PHI = torch.meshgrid(r, theta, phi, indexing='ij')
        
        actions_grid = torch.stack([
            R.flatten(),
            THETA.flatten(),
            PHI.flatten()
        ], dim=1)  # [N, 3]
        
        N = actions_grid.shape[0]
        actions_grid = actions_grid.unsqueeze(0).expand(B, -1, -1)  # [B, N, 3]

        # Get observation features
        obs_feat = self.obs_encoder(nobs)

        # Get energy for all grid points
        logits, coeffs = self.energy_head(obs_feat, actions_grid, return_coeffs=True)
        
        # Convert to probabilities
        action_probs = torch.softmax(logits / self.temperature, dim=-1)

        # Select action
        if self.sample_actions:
            flat_indexes = torch.multinomial(action_probs, num_samples=1).squeeze(-1)
        else:
            flat_indexes = torch.argmax(action_probs, dim=-1)

        # Get best action in spherical coords
        best_actions_spherical = actions_grid[torch.arange(B, device=device), flat_indexes]

        # Unnormalize radius
        r_normalized = best_actions_spherical[:, 0:1]
        dummy = torch.zeros(B, 1, 3, device=device)
        dummy[:, 0, 0] = r_normalized.squeeze()
        unnorm = self.normalizer["action"].unnormalize(dummy)
        r_unnorm = unnorm[:, 0, 0]

        theta_best = best_actions_spherical[:, 1]
        phi_best = best_actions_spherical[:, 2]

        # Convert to Cartesian
        x = r_unnorm * torch.sin(theta_best) * torch.cos(phi_best)
        y = r_unnorm * torch.sin(theta_best) * torch.sin(phi_best)
        z = r_unnorm * torch.cos(theta_best)

        actions = torch.stack([x, y, z], dim=1).unsqueeze(1)

        # Reshape energy for visualization
        energy_grid = logits.view(B, sbh.n_k, sbh.num_theta, sbh.num_phi)
        # print(f"=== Action Debug ===")
        # print(f"best_actions_spherical: {best_actions_spherical}")
        # print(f"r_unnorm: {r_unnorm}")
        # print(f"final action (xyz): {actions}")
        # print(f"=== End Action Debug ===")

        return {
            "action": actions,
            "energy": energy_grid,
            "fourier_coeffs": coeffs.cpu(),
        }

    def compute_loss(self, batch):
        """Compute InfoNCE loss."""
        # if not hasattr(self, '_norm_debug'):
        #     self._norm_debug = True
            
        #     print("\n=== Normalize Debug ===")
        #     print(f"batch obs keys: {batch['obs'].keys()}")
        #     print(f"batch keypoints shape: {batch['obs']['keypoints'].shape}")
        #     print(f"batch keypoints[0]: {batch['obs']['keypoints'][0]}")
        nobs = self.normalizer.normalize(batch["obs"])
        # print(f"\nnormalized keypoints shape: {nobs['keypoints'].shape}")
        # print(f"normalized keypoints[0]: {nobs['keypoints'][0]}")
        
        # print(f"\nbatch action shape: {batch['action'].shape}")
        # print(f"batch action[0]: {batch['action'][0]}")
        naction = self.normalizer["action"].normalize(batch["action"]).float()
        # print(f"normalized action[0]: {naction[0]}")
        # print("=== End Debug ===\n")

        To = self.num_obs_steps
        Ta = self.num_action_steps
        B = naction.shape[0]

        start = To - 1
        end = start + Ta
        naction = naction[:, start:end]

        noisy_actions = naction

        # Sample negatives
        action_stats = self.get_action_stats()
        action_dist = torch.distributions.Uniform(
            low=action_stats["min"], high=action_stats["max"]
        )

        negatives = action_dist.sample((B, self.num_neg_act_samples, Ta)).to(
            dtype=naction.dtype
        )

        # Combine pos and neg
        targets = torch.cat([noisy_actions.unsqueeze(1), negatives], dim=1)
        N = targets.size(1)

        # Permute
        permutation = torch.rand(B, N).argsort(dim=1)
        targets = targets[torch.arange(B).unsqueeze(-1), permutation]
        ground_truth = (permutation == 0).nonzero()[:, 1].to(naction.device)
        one_hot = F.one_hot(
            ground_truth, num_classes=self.num_neg_act_samples + 1
        ).float()

        # # Extract spherical coordinates
        # r = targets[:, :, 0, 0]
        # theta = self.normalizer["action"].unnormalize(targets)[:, :, 0, 1]
        # phi = self.normalizer["action"].unnormalize(targets)[:, :, 0, 2]
        # sphere_act = torch.concatenate(
        #     [r.view(B, N, 1), theta.view(B, N, 1), phi.view(B, N, 1)],
        #     axis=2,
        # )
        unnorm_targets = self.normalizer["action"].unnormalize(targets)
        sphere_act = unnorm_targets[:, :, 0, :]  # [B, N, 3] - (r, theta, phi)
        
        # Compute energy
        obs_feat = self.obs_encoder(nobs)
        energy = self.energy_head(obs_feat, sphere_act)

        # InfoNCE loss
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

    def plot_energy_fn(self, img, energy):
        """Plot energy function."""
        energy = torch.flip(energy, (1,))

        probs = torch.softmax(energy.view(1, -1) / self.temperature, dim=-1).view(
            self.energy_head.sbh.n_k,
            self.energy_head.sbh.num_theta,
            self.energy_head.sbh.num_phi,
        )

        fig = plt.figure()
        subfigs = fig.subfigures(1, 3)
        ax1 = subfigs[2].add_subplot()

        if img is not None:
            ax1.imshow(img[-1].transpose(1, 2, 0))
            ax1.set_title("Rollouts", va="bottom")
            ax1.set_axis_off()

        # Plot middle radius slice
        mid_r = self.energy_head.sbh.n_k // 2
        
        ax_e = subfigs[0].add_subplot()
        ax_e.imshow(energy[mid_r].cpu().numpy(), aspect='auto')
        ax_e.set_title(f"Energy (r={mid_r})")

        ax_p = subfigs[1].add_subplot()
        ax_p.imshow(probs[mid_r].cpu().numpy(), aspect='auto')
        ax_p.set_title(f"Prob (r={mid_r})")

        io_buf = io.BytesIO()
        fig.savefig(io_buf, format="raw")
        io_buf.seek(0)
        img_arr = np.reshape(
            np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
            newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
        )
        io_buf.close()
        plt.close()

        return img_arr