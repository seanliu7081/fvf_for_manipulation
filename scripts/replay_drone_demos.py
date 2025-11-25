#!/usr/bin/env python3
"""
Replay training demonstrations from drone_gt_100 dataset.

Usage:
    python scripts/replay_drone_demos.py --data_path drone_gt_100 --episode 0
    python scripts/replay_drone_demos.py --data_path drone_gt_100 --episode 0 --gui  # with GUI
    python scripts/replay_drone_demos.py --data_path drone_gt_100  # replay all episodes
"""

import os
import sys
import time
import numpy as np
import click

sys.path.insert(0, os.path.abspath("."))

from fvf.dataset.replay_buffer import ReplayBuffer
from fvf.env.drone.go_to_target_env import GoToTargetEnv
from fvf.env.drone.fly_through_gate_env import FlyThroughGateEnv


@click.command()
@click.option('-d', '--data_path', default='drone_gt_100', help='Path to the drone dataset')
@click.option('-e', '--episode', default=None, type=int, help='Episode number to replay (default: all)')
@click.option('-s', '--start_episode', default=0, type=int, help='Starting episode number')
@click.option('-n', '--num_episodes', default=None, type=int, help='Number of episodes to replay')
@click.option('--gui', is_flag=True, help='Enable PyBullet GUI visualization')
@click.option('--env_type', default='go_to_target', type=click.Choice(['go_to_target', 'fly_through_gate']), help='Environment type')
@click.option('--fps', default=5, type=int, help='Playback frames per second')
@click.option('--save_video', is_flag=True, help='Save videos of replays')
def main(data_path, episode, start_episode, num_episodes, gui, env_type, fps, save_video):
    """Replay drone demonstrations from a dataset."""
    
    # Load replay buffer
    print(f"Loading replay buffer from: {data_path}")
    replay_buffer = ReplayBuffer.create_from_path(data_path, mode='r')
    
    print(f"\nDataset Info:")
    print(f"  Total episodes: {replay_buffer.n_episodes}")
    print(f"  Total timesteps: {replay_buffer.n_steps}")
    print(f"  Episode lengths: min={np.min(replay_buffer.episode_lengths)}, "
          f"max={np.max(replay_buffer.episode_lengths)}, "
          f"mean={np.mean(replay_buffer.episode_lengths):.1f}")
    print(f"  Data keys: {list(replay_buffer.keys())}")
    
    # Determine which episodes to replay
    if episode is not None:
        episodes_to_replay = [episode]
    else:
        end_episode = start_episode + num_episodes if num_episodes else replay_buffer.n_episodes
        episodes_to_replay = range(start_episode, min(end_episode, replay_buffer.n_episodes))
    
    # Create environment
    if env_type == 'go_to_target':
        env = GoToTargetEnv(gui=gui, record=save_video)
    else:
        env = FlyThroughGateEnv(gui=gui, record=save_video)
    
    print(f"\nReplaying {len(episodes_to_replay)} episode(s)...")
    print(f"GUI: {gui}, FPS: {fps}, Save Video: {save_video}\n")
    
    # Replay episodes
    for ep_idx in episodes_to_replay:
        print(f"\n{'='*60}")
        print(f"Episode {ep_idx}")
        print(f"{'='*60}")
        
        # Get episode data
        episode_data = replay_buffer.get_episode(ep_idx)
        obs_data = episode_data['obs']  # Shape: (T, obs_dim)
        action_data = episode_data['action']  # Shape: (T, 3)
        
        episode_length = len(obs_data)
        print(f"  Length: {episode_length} steps")
        print(f"  Obs shape: {obs_data.shape}")
        print(f"  Action shape: {action_data.shape}")
        
        # Reset environment
        # Extract initial position and target from first observation
        initial_obs = obs_data[0]
        drone_pos = initial_obs[:3]  # First 3 elements
        target_pos = initial_obs[3:6]  # Next 3 elements (assuming go_to_target format)
        
        print(f"  Initial drone pos: [{drone_pos[0]:.3f}, {drone_pos[1]:.3f}, {drone_pos[2]:.3f}]")
        print(f"  Target pos: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        
        # Reset environment (will randomize, but we'll see the actions)
        env.reset(seed=ep_idx)
        
        # Replay actions
        total_reward = 0
        success = False
        
        sleep_time = 0.5 / fps
        
        for step_idx in range(episode_length):
            action = action_data[step_idx]
            
            # Step environment with recorded action
            obs, reward, done, terminated, info = env.step(action)
            total_reward += reward
            
            if reward > 0:
                success = True
            
            if gui:
                # Render will happen automatically in GUI mode
                pass
            else:
                # In non-GUI mode, print progress
                if step_idx % 10 == 0 or step_idx == episode_length - 1:
                    state = env._getDroneStateVector(0)
                    pos = state[:3]
                    print(f"  Step {step_idx:3d}/{episode_length}: "
                          f"pos=[{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}], "
                          f"reward={reward:.2f}")
            
            # Control playback speed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            if done or terminated:
                print(f"  Episode terminated at step {step_idx}")
                break
        
        print(f"\n  Results:")
        print(f"    Total reward: {total_reward:.2f}")
        print(f"    Success: {success}")
        print(f"    Final distance to target: {np.linalg.norm(env._getDroneStateVector(0)[:3] - env.TARGET_POS):.4f}")
    
    env.close()
    print(f"\n{'='*60}")
    print("Replay completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()