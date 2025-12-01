""" drone_image_runner.py - Runner for image-based drone policy evaluation """

import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import pybullet as pb
import wandb.sdk.data_types.video as wv
import imageio

from fvf.env.drone.go_to_target_env import GoToTargetEnv
from fvf.env.drone.fly_through_gate_env import FlyThroughGateEnv
from fvf.gym_util.async_vector_env import AsyncVectorEnv
from fvf.gym_util.multistep_wrapper import MultiStepWrapper
from fvf.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from fvf.gym_util.pybullet_video_wrapper import PyBulletVideoWrapper

from fvf.policy.base_policy import BasePolicy
from fvf.env_runner.base_runner import BaseRunner
from fvf.utils.torch_utils import dict_apply


def get_third_person_camera_image(env, width=96, height=96):
    """Get third-person view RGB image from PyBullet environment."""
    drone_state = env._getDroneStateVector(0)
    drone_pos = drone_state[:3]
    target_pos = env.TARGET_POS
    
    scene_center = (drone_pos + target_pos) / 2
    camera_distance = 1.5
    camera_pos = scene_center + np.array([0.5, 0.5, camera_distance])
    
    view_matrix = pb.computeViewMatrix(
        cameraEyePosition=camera_pos,
        cameraTargetPosition=scene_center,
        cameraUpVector=[0, 0, 1],
        physicsClientId=env.CLIENT
    )
    
    fov = 60
    aspect = width / height
    near = 0.01
    far = 100.0
    projection_matrix = pb.computeProjectionMatrixFOV(
        fov=fov, aspect=aspect, nearVal=near, farVal=far,
        physicsClientId=env.CLIENT
    )
    
    _, _, rgb, _, _ = pb.getCameraImage(
        width=width, height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=pb.ER_TINY_RENDERER,
        physicsClientId=env.CLIENT
    )
    
    rgb_image = np.array(rgb, dtype=np.uint8)[:, :, :3]
    return rgb_image


def get_keypoint_from_env(env):
    """Get keypoint [current_pos, initial_pos, target_pos] from environment."""
    drone_state = env._getDroneStateVector(0)
    current_pos = drone_state[:3]
    initial_pos = env.initial_pos if hasattr(env, 'initial_pos') and env.initial_pos is not None else current_pos
    target_pos = env.TARGET_POS
    
    keypoint = np.stack([current_pos, initial_pos, target_pos], axis=0)  # (3, 3)
    return keypoint.astype(np.float32)


class DroneImageRunner(BaseRunner):
    """Drone domain runner class for image-based policies."""

    def __init__(
        self,
        output_dir,
        env,
        keypoint_visible_rate=1.0,
        num_train=10,
        num_train_vis=1,
        train_start_seed=0,
        num_test=50,
        num_test_vis=2,
        test_start_seed=100000,
        max_steps=100,
        num_obs_steps=2,
        num_action_steps=1,
        num_latency_steps=0,
        fps=10,
        crf=22,
        agent_keypoints=False,
        past_action=False,
        tqdm_interval_sec=5.0,
        num_envs=None,
        action_coords="rectangular",
        render_size=96,
        crop_size=84,
    ):
        super().__init__(output_dir)
        num_envs = num_train + num_test if num_envs is None else num_envs
        self.action_coords = action_coords
        self.env_name = env
        self.render_size = render_size
        self.crop_size = crop_size

        env_num_obs_steps = num_obs_steps + num_latency_steps
        env_num_action_steps = num_action_steps

        if env == "go_to_target":
            self.env_class = GoToTargetEnv
        elif env == "fly_through_gate":
            self.env_class = FlyThroughGateEnv
        else:
            raise ValueError("Invalid env...")

        self.env_fns = []
        self.env_seeds = []
        self.env_prefixs = []
        self.env_init_fn_dills = []

        # Training seeds
        for i in range(num_train):
            seed = train_start_seed + i
            self.env_seeds.append(seed)
            self.env_prefixs.append("train/")

        # Testing seeds
        for i in range(num_test):
            seed = test_start_seed + i
            self.env_seeds.append(seed)
            self.env_prefixs.append("test/")

        self.fps = fps
        self.crf = crf
        self.agent_keypoints = agent_keypoints
        self.num_obs_steps = num_obs_steps
        self.num_action_steps = num_action_steps
        self.num_latency_steps = num_latency_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.num_train = num_train
        self.num_train_vis = num_train_vis
        self.num_test_vis = num_test_vis
        self.output_dir = output_dir

    def _get_obs_from_env(self, env, image_buffer, keypoint_buffer):
        """
        Get observation dict from environment.
        
        Args:
            env: GoToTargetEnv instance
            image_buffer: list of past images (T, H, W, C)
            keypoint_buffer: list of past keypoints (T, 3, 3)
        
        Returns:
            obs_dict: {"image": (1, T, C, H, W), "keypoint": (1, T, 3, 3)}
        """
        # Get current observation
        image = get_third_person_camera_image(env, self.render_size, self.render_size)
        keypoint = get_keypoint_from_env(env)
        
        # Add to buffer
        image_buffer.append(image)
        keypoint_buffer.append(keypoint)
        
        # Keep only last num_obs_steps
        while len(image_buffer) > self.num_obs_steps:
            image_buffer.pop(0)
        while len(keypoint_buffer) > self.num_obs_steps:
            keypoint_buffer.pop(0)
        
        # Pad if not enough history
        while len(image_buffer) < self.num_obs_steps:
            image_buffer.insert(0, image_buffer[0])
        while len(keypoint_buffer) < self.num_obs_steps:
            keypoint_buffer.insert(0, keypoint_buffer[0])
        
        # Stack: (T, H, W, C) -> (T, C, H, W)
        images = np.stack(image_buffer, axis=0)  # (T, H, W, C)
        images = np.moveaxis(images, -1, 1)  # (T, C, H, W)
        images = images.astype(np.float32) / 255.0
        
        keypoints = np.stack(keypoint_buffer, axis=0)  # (T, 3, 3)
        
        # Add batch dimension
        obs_dict = {
            "image": images[None, ...],      # (1, T, C, H, W)
            "keypoint": keypoints[None, ...], # (1, T, 3, 3)
        }
        
        return obs_dict

    def _run_single_env(self, global_idx, policy, device, record_video=False):
        """Run a single environment."""
        seed = self.env_seeds[global_idx]
        prefix = self.env_prefixs[global_idx]
        
        if record_video:
            print(f"[VIDEO] Recording environment {global_idx} ({prefix}seed={seed})")
        
        # Create environment
        base_env = self.env_class(gui=False, record=False)
        
        # Setup video recording if needed
        video_path = None
        video_frames = []
        
        if record_video:
            video_path = pathlib.Path(self.output_dir).joinpath(
                "media", wv.util.generate_id() + ".mp4"
            )
            video_path.parent.mkdir(parents=False, exist_ok=True)
            video_path = str(video_path)
        
        # Reset environment
        base_env.set_seed(seed)
        obs = base_env.reset(seed=seed)
        
        # Initialize observation buffers
        image_buffer = []
        keypoint_buffer = []
        
        episode_rewards = []
        episode_success = False
        policy.reset()
        
        pbar = tqdm.tqdm(
            total=self.max_steps,
            desc=f"{'Recording' if record_video else 'Eval'} {prefix}seed={seed}",
            leave=False,
            mininterval=self.tqdm_interval_sec,
        )
        
        step_count = 0
        while step_count < self.max_steps:
            # Get observation
            obs_dict = self._get_obs_from_env(base_env, image_buffer, keypoint_buffer)
            obs_dict = dict_apply(obs_dict, lambda x: torch.from_numpy(x).to(device))
            
            # Record frame for video
            if record_video:
                frame = get_third_person_camera_image(base_env, 256, 256)
                video_frames.append(frame)
            
            # Get action from policy
            with torch.no_grad():
                action_dict = policy.get_action(obs_dict, device)
            
            # Process action
            action = action_dict["action"][0, self.num_latency_steps:]
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            
            # Step environment
            obs, reward, done, truncated, info = base_env.step(action.flatten())
            
            # Collect reward
            try:
                r = float(reward) if np.isscalar(reward) else float(reward)
                episode_rewards.append(r)
            except:
                episode_rewards.append(0.0)
            
            step_count += 1
            pbar.update(1)
            
            if done or truncated:
                # Check success
                if isinstance(info, dict):
                    is_success_val = info.get('is_success', 0)
                    if isinstance(is_success_val, np.ndarray):
                        if len(is_success_val) > 0 and is_success_val[-1] > 0:
                            episode_success = True
                    elif is_success_val > 0:
                        episode_success = True
                break
        
        pbar.close()
        base_env.close()
        
        # Save video
        if record_video and video_frames:
            imageio.mimsave(video_path, video_frames, fps=self.fps)
            print(f"[VIDEO] Saved to {video_path}")
        
        return video_path, episode_rewards, episode_success

    def run(
        self,
        policy: BasePolicy,
        plot_energy_fn: bool = False,
        plot_weights_basis_fns: bool = False,
        use_break: bool = False,
    ):
        device = policy.device

        num_inits = len(self.env_seeds)
        all_video_paths = [None] * num_inits
        all_rewards = [None] * num_inits
        all_successes = [False] * num_inits

        for i in range(num_inits):
            prefix = self.env_prefixs[i]
            
            # Determine if video recording needed
            if prefix == "train/":
                env_idx = i
                record_video = env_idx < self.num_train_vis
            else:
                env_idx = i - self.num_train
                record_video = env_idx < self.num_test_vis
            
            # Run environment
            video_path, rewards, success = self._run_single_env(
                i, policy, device, record_video=record_video
            )
            
            all_video_paths[i] = video_path
            all_rewards[i] = rewards
            all_successes[i] = success

        # Logging
        max_rewards = collections.defaultdict(list)
        success_list = collections.defaultdict(list)
        log_data = dict()

        for i in range(num_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            
            if all_rewards[i] is not None and len(all_rewards[i]) > 0:
                total_reward = np.sum(all_rewards[i])
            else:
                total_reward = 0.0
            
            is_success = 1.0 if all_successes[i] else 0.0
            
            max_rewards[prefix].append(total_reward)
            success_list[prefix].append(is_success)
            log_data[prefix + f"sim_total_reward_{seed}"] = total_reward
            log_data[prefix + f"sim_success_{seed}"] = is_success

            # Log video
            video_path = all_video_paths[i]
            if video_path is not None:
                import os
                if os.path.exists(video_path):
                    try:
                        sim_video = wandb.Video(video_path)
                        log_data[prefix + f"sim_video_{seed}"] = sim_video
                        print(f"[VIDEO] Logged to WandB: {prefix}sim_video_{seed}")
                    except Exception as e:
                        print(f"[VIDEO] Failed to log to WandB: {e}")

        # Log aggregate metrics
        for prefix, v in max_rewards.items():
            mean_reward = np.mean(v)
            success_count = sum(success_list[prefix])
            total_count = len(success_list[prefix])
            mean_success = np.mean(success_list[prefix])
            log_data[prefix + "mean_score"] = mean_success
            log_data[prefix + "mean_reward"] = mean_reward
            print(f"{prefix}mean_score (success rate) = {mean_success:.3f} ({int(success_count)}/{total_count} succeeded)")
            print(f"{prefix}mean_reward = {mean_reward:.3f}")

        return log_data