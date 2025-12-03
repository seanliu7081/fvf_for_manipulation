# import numpy as np
# import torch
# from typing import Dict

# from fvf.dataset.base_dataset import BaseDataset
# from fvf.utils import normalize_utils, data_augmentation, action_utils, torch_utils
# from fvf.model.common.normalizer import SingleFieldLinearNormalizer


# class DroneImageDataset(BaseDataset):
#     """
#     Dataset for drone tasks with image + keypoints observations.
    
#     Data format (from zarr):
#         - img: (T, 96, 96, 3) uint8
#         - keypoints: (T, 3, 3) float32 - [current_pos, initial_pos, target_pos]
#         - action: (T, 3) float32
    
#     Output format (for DroneObsEncoder with use_keypoints=True):
#         - obs["image"]: (T, C, H, W) float32
#         - obs["keypoints"]: (T, 3, 3) float32
#         - action: (T, 3) float32
#     """
#     def __init__(
#         self,
#         path,
#         horizon=1,
#         pad_before=0,
#         pad_after=0,
#         action_coords: str = "rectangular",
#         crop_image_size: int = 84,
#         seed=0,
#         val_ratio=0.0,
#         max_train_episodes=None,
#     ):
#         self.crop_image_size = crop_image_size

#         super().__init__(
#             path,
#             horizon=horizon,
#             pad_before=pad_before,
#             pad_after=pad_after,
#             buffer_keys=['img', 'keypoints', 'action'],
#             action_coords=action_coords,
#             seed=seed,
#             val_ratio=val_ratio,
#             max_train_episodes=max_train_episodes
#         )

#     def get_normalizer(self, mode="limits", **kwargs):
#         sample_data = self._sample_to_data(self.replay_buffer, rand_crop=False)
        
#         # Flatten keypoints: (T, 3, 3) -> (T, 9)
#         keypoints_flat = sample_data['obs']['keypoints'].reshape(-1, 9)
        
#         data = {
#             'action': sample_data['action'],
#             'keypoints': keypoints_flat,
#         }
#         data["action"] = action_utils.convert_action_coords(data["action"], self.action_coords)

#         normalizer = super().get_normalizer(data, mode=mode, **kwargs)

#         # Action normalizer
#         act_norm = SingleFieldLinearNormalizer()
#         act_norm.fit(data=data['action'], output_min=-1, output_max=1)
#         normalizer['action'] = act_norm

#         # keypoints normalizer (3 points × 3D = 9 dim)
#         normalizer['keypoints'] = self._keypoints_normalizer(keypoints_flat)

#         # Image normalizer
#         normalizer['image'] = normalize_utils.get_image_range_normalizer()

#         return normalizer

#     def _keypoints_normalizer(self, arr, nmin=-1.0, nmax=1.0):
#         """Normalizer for 3D keypointss. Workspace: x,y in [-1.5, 1.5], z in [0, 1.5]"""
#         stat = {
#             "min": np.array([-1.5, -1.5, 0.0] * 3, dtype=np.float32),
#             "max": np.array([1.5, 1.5, 1.5] * 3, dtype=np.float32),
#             "mean": np.mean(arr, axis=0),
#             "std": np.std(arr, axis=0) + 1e-6,
#         }
#         scale = (nmax - nmin) / (stat["max"] - stat["min"])
#         offset = nmin - scale * stat["min"]
#         return SingleFieldLinearNormalizer.create_manual(
#             scale=scale, offset=offset, input_stats_dict=stat
#         )

#     def _sample_to_data(self, sample, rand_crop=True):
#         T = sample['img'].shape[0]

#         # Image: (T, H, W, C) -> (T, C, H, W), normalize to [0, 1]
#         obs_img = np.moveaxis(sample['img'], -1, 1) / 255.0
        
#         if rand_crop:
#             obs_img = data_augmentation.random_crop(obs_img, self.crop_image_size)

#         # keypoints: (T, 3, 3)
#         keypoints = sample['keypoints'].reshape(T, 3, 3)

#         # Action: (T, 3)
#         action = sample['action'].reshape(T, 3)

#         data = {
#             'obs': {
#                 'image': obs_img.astype(np.float32),
#                 'keypoints': keypoints.astype(np.float32),
#             },
#             'action': action.astype(np.float32),
#         }
#         return data
    
#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         sample = self.sampler.sample_sequence(idx)
#         data = self._sample_to_data(sample, rand_crop=True)
#         data["action"] = action_utils.convert_action_coords(data["action"], self.action_coords)

#         torch_data = torch_utils.dict_apply(data, torch.from_numpy)
#         return torch_data


import numpy as np
import torch
from typing import Dict

from fvf.dataset.base_dataset import BaseDataset
from fvf.utils import normalize_utils, data_augmentation, action_utils, torch_utils
from fvf.model.common.normalizer import SingleFieldLinearNormalizer


class DroneImageDataset(BaseDataset):
    """
    Dataset for drone tasks with image + keypoints observations.
    
    Data format (from zarr):
        - img: (T, 96, 96, 3) uint8
        - keypoints: (T, 3, 3) float32 - [current_pos, initial_pos, target_pos]
        - action: (T, 3) float32
    
    Output format (for DroneObsEncoder with use_keypoints=True):
        - obs["image"]: (T, C, H, W) float32
        - obs["keypoints"]: (T, 3, 3) float32
        - action: (T, 3) float32
    """
    def __init__(
        self,
        path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        action_coords: str = "rectangular",
        crop_image_size: int = 84,
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
        sample_data = self._sample_to_data(self.replay_buffer, rand_crop=False)
        
        # Flatten keypoints: (T, 3, 3) -> (T, 9)
        keypoints_flat = sample_data['obs']['keypoints'].reshape(-1, 9)
        
        data = {
            'action': sample_data['action'],
            'keypoints': keypoints_flat,
        }
        data["action"] = action_utils.convert_action_coords(data["action"], self.action_coords)

        normalizer = super().get_normalizer(data, mode=mode, **kwargs)

        # Action normalizer - use same normalizer as DroneDataset for consistency
        if self.action_coords == "spherical":
            normalizer['action'] = self._act_normalizer(data['action'])
        else:
            act_norm = SingleFieldLinearNormalizer()
            act_norm.fit(data=data['action'], output_min=-1, output_max=1)
            normalizer['action'] = act_norm

        # Keypoints normalizer (3 points × 3D = 9 dim)
        normalizer['keypoints'] = self._keypoints_normalizer(keypoints_flat)

        # Image normalizer
        normalizer['image'] = normalize_utils.get_image_range_normalizer()

        return normalizer

    def _act_normalizer(self, arr, nmin=0.0, nmax=1.0):
        """
        Action normalizer for spherical coordinates.
        Must match DroneDataset's act_normalizer for training/inference consistency.
        
        Spherical coords: (r, theta, phi)
            - r: [0, max_r] radius
            - theta: [0, pi] polar angle
            - phi: [0, 2*pi] azimuthal angle
        
        Output range: [0, 1] (not [-1, 1]!)
        """
        stat = {
            "min": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "max": np.array([arr.max(0)[0], np.pi, 2 * np.pi], dtype=np.float32),
            "mean": np.mean(arr, axis=0),
            "std": np.std(arr, axis=0),
        }
        scale = (nmax - nmin) / (stat["max"] - stat["min"])
        offset = nmin - scale * stat["min"]
        return SingleFieldLinearNormalizer.create_manual(
            scale=scale, offset=offset, input_stats_dict=stat
        )

    def _keypoints_normalizer(self, arr, nmin=-1.0, nmax=1.0):
        """
        Normalizer for 3D keypoints.
        Workspace bounds: x,y in [-1.5, 1.5], z in [0, 1.5]
        """
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

    def _sample_to_data(self, sample, rand_crop=True):
        T = sample['img'].shape[0]

        # Image: (T, H, W, C) -> (T, C, H, W), normalize to [0, 1]
        obs_img = np.moveaxis(sample['img'], -1, 1) / 255.0
        
        if rand_crop:
            obs_img = data_augmentation.random_crop(obs_img, self.crop_image_size)

        # Keypoints: (T, 3, 3)
        keypoints = sample['keypoints'].reshape(T, 3, 3)

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
        data = self._sample_to_data(sample, rand_crop=True)
        data["action"] = action_utils.convert_action_coords(data["action"], self.action_coords)

        torch_data = torch_utils.dict_apply(data, torch.from_numpy)
        return torch_data