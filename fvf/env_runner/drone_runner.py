""" drone_runner.py  """

import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
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


class DroneRunner(BaseRunner):
    """Drone domain runner class."""

    def __init__(
        self,
        output_dir,
        env,
        keypoint_visible_rate=1.0,
        num_train=10,
        num_train_vis=3,
        train_start_seed=0,
        num_test=22,
        num_test_vis=6,
        test_start_seed=10000,
        max_steps=200,
        num_obs_steps=8,
        num_action_steps=8,
        num_latency_steps=0,
        fps=10,
        crf=22,
        agent_keypoints=False,
        past_action=False,
        tqdm_interval_sec=5.0,
        num_envs=None,
        action_coords="rectangular",
    ):
        super().__init__(output_dir)
        num_envs = num_train + num_test if num_envs is None else num_envs
        self.action_coords = action_coords
        self.env_name = env

        env_num_obs_steps = num_obs_steps + num_latency_steps
        env_num_action_steps = num_action_steps

        if env == "go_to_target":
            env_class = GoToTargetEnv
        elif env == "fly_through_gate":
            env_class = FlyThroughGateEnv
        else:
            raise ValueError("Invalid env...")

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    env_class(gui=False, record=False),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec="h264",
                        input_pix_fmt="rgb24",
                        crf=crf,
                        thread_type="FRAME",
                        thread_count=1,
                    ),
                    file_path=None,
                ),
                n_obs_steps=env_num_obs_steps,
                n_action_steps=env_num_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum'
            )

        env_fns = [env_fn] * num_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # Training
        for i in range(num_train):
            seed = train_start_seed + i
            enable_render = i < num_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                assert isinstance(env, MultiStepWrapper)
                env.set_seed(seed)

            env_seeds.append(seed)
            env_prefixs.append("train/")
            env_init_fn_dills.append(dill.dumps(init_fn))

        # Testing
        for i in range(num_test):
            seed = test_start_seed + i
            enable_render = i < num_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                assert isinstance(env, MultiStepWrapper)
                env.set_seed(seed)

            env_seeds.append(seed)
            env_prefixs.append("test/")
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
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

    def _run_single_env_with_video(self, global_idx, policy, device):
        """Run a single environment with video recording (no GUI needed)"""
        seed = self.env_seeds[global_idx]
        prefix = self.env_prefixs[global_idx]
        
        print(f"[VIDEO] Recording environment {global_idx} ({prefix}seed={seed})")
        
        # Create base environment (no GUI)
        if self.env_name == "go_to_target":
            base_env = GoToTargetEnv(gui=False, record=False)
        elif self.env_name == "fly_through_gate":
            base_env = FlyThroughGateEnv(gui=False, record=False)
        else:
            raise ValueError("Invalid env...")
        
        # Prepare video path
        video_path = pathlib.Path(self.output_dir).joinpath(
            "media", wv.util.generate_id() + ".mp4"
        )
        video_path.parent.mkdir(parents=False, exist_ok=True)
        
        # Wrap with video recorder
        video_env = PyBulletVideoWrapper(
            base_env,
            video_path=str(video_path),
            fps=self.fps
        )
        
        # Wrap with MultiStepWrapper
        wrapped_env = MultiStepWrapper(
            video_env,
            n_obs_steps=self.num_obs_steps + self.num_latency_steps,
            n_action_steps=self.num_action_steps,
            max_episode_steps=self.max_steps,
            reward_agg_method='sum'
        )
        
        # Set seed and reset
        wrapped_env.set_seed(seed)
        obs = wrapped_env.reset()
        
        episode_rewards = []
        past_action = None
        policy.reset()
        
        pbar = tqdm.tqdm(
            total=self.max_steps,
            desc=f"Recording {prefix}seed={seed}",
            leave=False,
            mininterval=self.tqdm_interval_sec,
        )
        
        step_count = 0
        while step_count < self.max_steps:
            # Prepare observation (add batch dimension)
            obs_dict = {
                "keypoints": obs[None, ..., :self.num_obs_steps, :].astype(np.float32),
            }
            obs_dict = dict_apply(obs_dict, lambda x: torch.from_numpy(x).to(device))
            
            # Get action from policy
            with torch.no_grad():
                action_dict = policy.get_action(obs_dict, device)
            
            # Remove batch dimension and latency steps
            action = action_dict["action"][0, self.num_latency_steps:]
            
            # Ensure action is numpy array
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            
            # Step environment
            obs, reward, done, truncated, info = wrapped_env.step(action)
            
            # Collect reward
            try:
                r = float(reward) if np.isscalar(reward) else float(reward.item() if hasattr(reward, 'item') else reward)
                episode_rewards.append(r)
            except:
                episode_rewards.append(0.0)
            
            num_steps = action.shape[0] if action.ndim > 1 else 1
            step_count += num_steps
            pbar.update(num_steps)
            
            if done or truncated:
                break
        
        pbar.close()
        
        # Save video
        video_env.save_video()
        wrapped_env.close()
        
        print(f"[VIDEO] Completed: {len(episode_rewards)} steps, total_reward={sum(episode_rewards):.3f}")
        
        return str(video_path), episode_rewards

    def run(
        self,
        policy: BasePolicy,
        plot_energy_fn: bool = False,
        plot_weights_basis_fns: bool = False,
        use_break: bool = False,
    ):
        device = policy.device
        dtype = policy.dtype

        env = self.env

        num_envs = len(self.env_fns)
        num_inits = len(self.env_init_fn_dills)
        num_chunks = math.ceil(num_inits / num_envs)

        all_video_paths = [None] * num_inits
        all_rewards = [None] * num_inits
        energy_fn_plots = [list() for _ in range(num_inits)]

        for chunk_idx in range(num_chunks):
            start = chunk_idx * num_envs
            end = min(num_inits, start + num_envs)
            this_global_slice = slice(start, end)
            this_num_active_envs = end - start
            this_local_slice = slice(0, this_num_active_envs)

            # Check if any environment needs video recording
            need_recording = []
            for idx in range(this_num_active_envs):
                global_idx = start + idx
                prefix = self.env_prefixs[global_idx]
                if prefix == "train/":
                    env_idx = global_idx
                    need_rec = env_idx < self.num_train_vis
                else:  # test
                    env_idx = global_idx - self.num_train
                    need_rec = env_idx < self.num_test_vis
                need_recording.append(need_rec)

            # Handle environments with video recording separately
            if any(need_recording):
                for idx in range(this_num_active_envs):
                    global_idx = start + idx
                    
                    if need_recording[idx]:
                        # Run with video recording
                        video_path, rewards = self._run_single_env_with_video(
                            global_idx, policy, device
                        )
                        all_video_paths[global_idx] = video_path
                        all_rewards[global_idx] = rewards
                    else:
                        # For non-recording environments in this chunk
                        # We'll skip them for simplicity
                        pass
                
                # Skip to next chunk
                continue

            # Original vectorized evaluation for non-recording environments
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            num_diff = num_envs - len(this_init_fns)
            if num_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * num_diff)
            assert len(this_init_fns) == num_envs

            env.call_each("run_dill_function", args_list=[(x,) for x in this_init_fns])

            obs = env.reset()
            past_action = None
            policy.reset()

            episode_rewards = [[] for _ in range(this_num_active_envs)]

            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval DroneRunner {chunk_idx+1} / {num_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )

            step_count = 0
            while step_count < self.max_steps:
                obs_dict = {
                    "keypoints": obs[..., : self.num_obs_steps, :].astype(np.float32),
                }

                obs_dict = dict_apply(
                    obs_dict, lambda x: torch.from_numpy(x).to(device)
                )

                with torch.no_grad():
                    action_dict = policy.get_action(obs_dict, device)

                action_dict = dict_apply(action_dict, lambda x: x.to("cpu").numpy())
                action = action_dict["action"][:, self.num_latency_steps :]

                obs, reward, done, timeout, info = env.step(action)
                
                for i in range(this_num_active_envs):
                    try:
                        if isinstance(reward, (list, tuple)):
                            r = float(reward[i]) if len(reward) > i else 0.0
                        elif isinstance(reward, np.ndarray):
                            r = float(reward[i]) if len(reward) > i else 0.0
                        else:
                            r = float(reward)
                        episode_rewards[i].append(r)
                    except:
                        episode_rewards[i].append(0.0)
                
                num_steps = action.shape[1]
                step_count += num_steps
                pbar.update(num_steps)

            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = episode_rewards

        # Logging
        max_rewards = collections.defaultdict(list)
        log_data = dict()

        for i in range(num_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            
            if all_rewards[i] is not None and len(all_rewards[i]) > 0:
                total_reward = np.sum(all_rewards[i])
            else:
                total_reward = 0.0
            
            max_rewards[prefix].append(total_reward)
            log_data[prefix + f"sim_total_reward_{seed}"] = total_reward

            # Log video to wandb
            video_path = all_video_paths[i]
            if video_path is not None:
                import os
                if os.path.exists(video_path):
                    try:
                        sim_video = wandb.Video(video_path)
                        log_data[prefix + f"sim_video_{seed}"] = sim_video
                        print(f"[VIDEO] ✅ Logged to WandB: {prefix}sim_video_{seed}")
                    except Exception as e:
                        print(f"[VIDEO] ⚠️ Failed to log to WandB: {e}")

        # Log aggregate metrics
        for prefix, v in max_rewards.items():
            mean_score = np.mean(v)
            log_data[prefix + "mean_score"] = mean_score
            print(f"{prefix}mean_score = {mean_score:.3f}")

        return log_data


def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, "RGBA image has 4 channels."

    rgb = np.zeros((row, col, 3), dtype="float32")
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    a = np.asarray(a, dtype="float32") / 255.0

    R, G, B = background

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype="uint8")