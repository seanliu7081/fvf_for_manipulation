"""
spherical_bessel_implicit_rd_policy.py - RD version
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
from fvf.utils import mcmc


class SphericalBesselImplicitRdPolicy(BasePolicy):

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
        action_sampling: str = 'dfo',
        pred_n_iter: int = 3,
        optimize_negatives: bool = False,
        sample_actions: bool = False,
        temperature: float = 1.0,
        grad_pen: bool = False,
    ):
        super().__init__(obs_dim, action_dim, num_obs_steps, num_action_steps)
        self.action_sampling = action_sampling
        self.pred_n_iter = pred_n_iter
        self.num_neg_act_samples = num_neg_act_samples
        self.pred_n_samples = pred_n_samples
        self.optimize_negatives = optimize_negatives
        self.sample_actions = sample_actions
        self.temperature = temperature
        self.grad_pen = grad_pen

        self.obs_encoder = obs_encoder
        self.energy_head = energy_head
        self.apply(torch_utils.init_weights)

    def get_action(self, obs, device, use_break=False):
        nobs = self.normalizer.normalize(obs)
        B = list(obs.values())[0].shape[0]

        Ta = self.num_action_steps
        
        action_stats = self.get_action_stats()
        action_dist = torch.distributions.Uniform(
            low=action_stats["min"], high=action_stats["max"]
        )

        # Sample initial actions
        actions = action_dist.sample((B, self.pred_n_samples, Ta))

        # Get observation features
        obs_feat = self.obs_encoder(nobs)

        # Use DFO to optimize actions
        if self.action_sampling == "dfo":
            action_probs, actions = mcmc.iterative_dfo(
                self.energy_head,
                obs_feat,
                actions,
                [action_stats["min"], action_stats["max"]],
                normalizer=self.normalizer,
            )
        elif self.action_sampling == "langevin":
            action_probs, actions = mcmc.langevin_actions(
                self.energy_head,
                obs_feat,
                actions,
                [action_stats["min"], action_stats["max"]],
                num_iterations=100,
                normalizer=self.normalizer,
            )
        else:
            logits = self.energy_head(obs_feat, actions)
            action_probs = torch.softmax(logits / self.temperature, dim=-1)

        # Select action
        if self.sample_actions:
            idxs = torch.multinomial(action_probs, num_samples=1, replacement=True)
        else:
            idxs = torch.argmax(action_probs, dim=-1)
        
        actions = actions[torch.arange(B).unsqueeze(-1), idxs].squeeze(1)
        actions = self.normalizer["action"].unnormalize(actions)

        return {"action": actions}

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch["obs"])
        naction = self.normalizer["action"].normalize(batch["action"]).float()

        To = self.num_obs_steps
        Ta = self.num_action_steps
        B = naction.shape[0]

        start = To - 1
        end = start + Ta
        naction = naction[:, start:end]

        # Add noise to positive samples
        action_noise = torch.normal(
            mean=0,
            std=1e-4,
            size=naction.shape,
            dtype=naction.dtype,
            device=naction.device,
        )
        noisy_actions = naction + action_noise

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

        # Compute energy - energy_head
        obs_feat = self.obs_encoder(nobs)
        energy = self.energy_head(obs_feat, targets)

        # InfoNCE loss
        probs = F.log_softmax(energy, dim=1)
        ebm_loss = F.kl_div(probs, one_hot, reduction="batchmean")

        return ebm_loss, ebm_loss, torch.tensor(0.0), torch.tensor(0.0)

    def get_action_stats(self):
        action_stats = self.normalizer["action"].get_output_stats()
        repeated_stats = dict()
        for key, value in action_stats.items():
            n_repeats = self.action_dim // value.shape[0]
            repeated_stats[key] = value.repeat(n_repeats)
        return repeated_stats