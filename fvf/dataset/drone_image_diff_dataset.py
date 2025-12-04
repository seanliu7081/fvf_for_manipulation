# fvf/dataset/drone_image_diff_dataset.py

import numpy as np
import torch
from typing import Dict

from fvf.dataset.base_dataset import BaseDataset
from fvf.utils import normalize_utils, action_utils, torch_utils
from fvf.model.common.normalizer import SingleFieldLinearNormalizer


class DroneImageDiffDataset(BaseDataset):
    """
    Dataset for drone tasks with Diffusion Policy.
    
    Key differences from DroneImageDataset:
    - No cropping (MultiImageObsEncoder handles it)
    - Keypoints output as (T, 9) not (T, 3, 3)
    - Action normalizer uses [-1, 1] range for rectangular coords
    """
    def __init__(
        self,
        path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        action_coords: str = "rectangular",
        crop_image_size: int = 84,  # not used here, kept for config compatibility
        seed=0,
        val_ratio=0.0,
        max_train_episodes=None,
    ):
        self.crop_image_size = crop_image_size

        super().__init__(
            path,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            buffer_keys=['img', 'keypoints', 'action'],
            action_coords=action_coords,
            seed=seed,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes
        )

    def get_normalizer(self, mode="limits", **kwargs):
        sample_data = self._sample_to_data(self.replay_buffer)
        
        data = {
            'action': sample_data['action'],
            'keypoints': sample_data['obs']['keypoints'],  # already (T, 9)
        }
        data["action"] = action_utils.convert_action_coords(data["action"], self.action_coords)

        normalizer = super().get_normalizer(data, mode=mode, **kwargs)

        # Action normalizer: [-1, 1] for diffusion policy
        act_norm = SingleFieldLinearNormalizer()
        act_norm.fit(data=data['action'], output_min=-1, output_max=1)
        normalizer['action'] = act_norm

        # Keypoints normalizer
        normalizer['keypoints'] = self._keypoints_normalizer(data['keypoints'])

        # Image normalizer
        normalizer['image'] = normalize_utils.get_image_range_normalizer()

        return normalizer

    def _keypoints_normalizer(self, arr, nmin=-1.0, nmax=1.0):
        """Normalizer for 3D keypoints. Workspace: x,y in [-1.5, 1.5], z in [0, 1.5]"""
        stat = {
            "min": np.array([-1.5, -1.5, 0.0] * 3, dtype=np.float32),
            "max": np.array([1.5, 1.5, 1.5] * 3, dtype=np.float32),
            "mean": np.mean(arr, axis=0),
            "std": np.std(arr, axis=0) + 1e-6,
        }
        scale = (nmax - nmin) / (stat["max"] - stat["min"])
        offset = nmin - scale * stat["min"]
        return SingleFieldLinearNormalizer.create_manual(
            scale=scale, offset=offset, input_stats_dict=stat
        )

    def _sample_to_data(self, sample):
        T = sample['img'].shape[0]

        # Image: (T, H, W, C) -> (T, C, H, W), normalize to [0, 1]
        # NO CROPPING - let MultiImageObsEncoder handle it
        obs_img = np.moveaxis(sample['img'], -1, 1) / 255.0

        # Keypoints: (T, 3, 3) -> (T, 9) for MultiImageObsEncoder
        keypoints = sample['keypoints'].reshape(T, 9)

        # Action: (T, 3)
        action = sample['action'].reshape(T, 3)

        data = {
            'obs': {
                'image': obs_img.astype(np.float32),
                'keypoints': keypoints.astype(np.float32),
            },
            'action': action.astype(np.float32),
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        data["action"] = action_utils.convert_action_coords(data["action"], self.action_coords)

        torch_data = torch_utils.dict_apply(data, torch.from_numpy)
        return torch_data