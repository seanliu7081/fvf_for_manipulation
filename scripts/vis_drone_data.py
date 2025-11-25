import zarr
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(".."))

from fvf.dataset.replay_buffer import ReplayBuffer

# Path to your data
data_path = "/media/lht/T7TwoTB/code/fourier_value_functions/drone_gt_100"  # or the full path you used

# Load the replay buffer
print(f"Loading data from: {data_path}")
replay_buffer = ReplayBuffer.create_from_path(data_path, mode='r')

# Print the structure
print("\n" + "="*60)
print("REPLAY BUFFER STRUCTURE")
print("="*60)
print(replay_buffer)

# Print basic statistics
print("\n" + "="*60)
print("BASIC STATISTICS")
print("="*60)
print(f"Total episodes: {replay_buffer.n_episodes}")
print(f"Total timesteps: {replay_buffer.n_steps}")
print(f"Episode lengths: {replay_buffer.episode_lengths}")
print(f"Mean episode length: {np.mean(replay_buffer.episode_lengths):.2f}")
print(f"Min episode length: {np.min(replay_buffer.episode_lengths)}")
print(f"Max episode length: {np.max(replay_buffer.episode_lengths)}")

# Print data keys and shapes
print("\n" + "="*60)
print("DATA KEYS AND SHAPES")
print("="*60)
for key in replay_buffer.keys():
    data = replay_buffer[key]
    print(f"{key:20s}: shape={data.shape}, dtype={data.dtype}")

# Examine a single episode
print("\n" + "="*60)
print("FIRST EPISODE DETAILS")
print("="*60)
episode_0 = replay_buffer.get_episode(0)
for key, value in episode_0.items():
    print(f"\n{key}:")
    print(f"  Shape: {value.shape}")
    print(f"  Dtype: {value.dtype}")
    print(f"  Min: {np.min(value):.4f}, Max: {np.max(value):.4f}, Mean: {np.mean(value):.4f}")
    print(f"  First timestep: {value[0]}")
    print(f"  Last timestep: {value[-1]}")

# Check if images are present
print("\n" + "="*60)
print("DATA CONTENT CHECK")
print("="*60)
has_images = any('img' in key or 'image' in key for key in replay_buffer.keys())
has_states = any('obs' in key or 'state' in key for key in replay_buffer.keys())
has_actions = 'action' in replay_buffer.keys()

print(f"Contains images: {has_images}")
print(f"Contains states/observations: {has_states}")
print(f"Contains actions: {has_actions}")

print("\n" + "="*60)
print("METADATA")
print("="*60)
for key, value in replay_buffer.meta.items():
    print(f"{key}: {value}")