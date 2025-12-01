#!/usr/bin/env python3
"""
Visualization script for pusht_demo.zarr data
Usage: python vis_pusht_data.py [--episode EPISODE] [--save SAVE_PATH]
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from fvf.dataset.replay_buffer import ReplayBuffer


def print_statistics(replay_buffer):
    """Print detailed statistics about the replay buffer"""
    print("\n" + "="*60)
    print("REPLAY BUFFER STRUCTURE")
    print("="*60)
    print(replay_buffer)
    
    print("\n" + "="*60)
    print("BASIC STATISTICS")
    print("="*60)
    print(f"Total episodes: {replay_buffer.n_episodes}")
    print(f"Total timesteps: {replay_buffer.n_steps}")
    print(f"Episode lengths: {replay_buffer.episode_lengths}")
    if len(replay_buffer.episode_lengths) > 0:
        print(f"Mean episode length: {np.mean(replay_buffer.episode_lengths):.2f}")
        print(f"Min episode length: {np.min(replay_buffer.episode_lengths)}")
        print(f"Max episode length: {np.max(replay_buffer.episode_lengths)}")
    
    print("\n" + "="*60)
    print("DATA KEYS AND SHAPES")
    print("="*60)
    for key in replay_buffer.keys():
        data = replay_buffer[key]
        print(f"{key:20s}: shape={data.shape}, dtype={data.dtype}")
    
    print("\n" + "="*60)
    print("DATA CONTENT CHECK")
    print("="*60)
    has_images = any('img' in key for key in replay_buffer.keys())
    has_states = any('state' in key for key in replay_buffer.keys())
    has_keypoints = any('keypoint' in key for key in replay_buffer.keys())
    has_actions = 'action' in replay_buffer.keys()
    has_contacts = any('contact' in key for key in replay_buffer.keys())
    
    print(f"Contains images: {has_images}")
    print(f"Contains states: {has_states}")
    print(f"Contains keypoints: {has_keypoints}")
    print(f"Contains actions: {has_actions}")
    print(f"Contains contacts: {has_contacts}")


def visualize_episode(replay_buffer, episode_idx, save_path=None):
    """Visualize a single episode with images, keypoints, states, and actions"""
    if episode_idx >= replay_buffer.n_episodes:
        print(f"Episode {episode_idx} does not exist. Total episodes: {replay_buffer.n_episodes}")
        return
    
    episode = replay_buffer.get_episode(episode_idx)
    print(f"\nVisualizing episode {episode_idx}...")
    
    # Get data
    images = episode.get('img', None)
    states = episode.get('state', None)
    keypoints = episode.get('keypoint', None)
    actions = episode.get('action', None)
    n_contacts = episode.get('n_contacts', None)
    
    if images is None:
        print("No images found in episode!")
        return
    
    T = images.shape[0]
    print(f"Episode length: {T} timesteps")
    
    # Convert image format: (T, H, W, C) -> (T, C, H, W) for display
    if len(images.shape) == 4 and images.shape[-1] == 3:
        # Images are in (T, H, W, C) format
        images_display = images
    elif len(images.shape) == 4 and images.shape[1] == 3:
        # Images are in (T, C, H, W) format
        images_display = np.moveaxis(images, 1, -1)
    else:
        images_display = images
    
    # Normalize images to [0, 1] if needed
    if images_display.dtype == np.uint8:
        images_display = images_display.astype(np.float32) / 255.0
    elif images_display.max() > 1.0:
        images_display = images_display / 255.0
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main image display
    ax_img = fig.add_subplot(gs[0:2, 0:2])
    ax_img.set_title(f'Episode {episode_idx} - Frame 0/{T-1}', fontsize=14)
    ax_img.axis('off')
    
    # Keypoints overlay (if available)
    if keypoints is not None:
        ax_kp = fig.add_subplot(gs[0:2, 0:2])
        ax_kp.axis('off')
        ax_kp.set_xlim(0, images_display.shape[2])
        ax_kp.set_ylim(images_display.shape[1], 0)
    
    # State plot
    ax_state = fig.add_subplot(gs[0, 2])
    ax_state.set_title('State (Agent + Block)', fontsize=12)
    
    # Action plot
    ax_action = fig.add_subplot(gs[1, 2])
    ax_action.set_title('Actions', fontsize=12)
    ax_action.set_xlabel('Timestep')
    ax_action.set_ylabel('Action Value')
    
    # Contacts plot (if available)
    ax_contact = None
    if n_contacts is not None:
        ax_contact = fig.add_subplot(gs[2, 0])
        ax_contact.set_title('Number of Contacts', fontsize=12)
        ax_contact.set_xlabel('Timestep')
        ax_contact.set_ylabel('Contacts')
    
    # Keypoints trajectory plot
    ax_kp_traj = None
    if keypoints is not None:
        ax_kp_traj = fig.add_subplot(gs[2, 1])
        ax_kp_traj.set_title('Keypoint Trajectory', fontsize=12)
        ax_kp_traj.set_xlabel('X')
        ax_kp_traj.set_ylabel('Y')
        ax_kp_traj.set_aspect('equal')
    
    # Statistics text
    ax_stats = fig.add_subplot(gs[2, 2])
    ax_stats.axis('off')
    
    # Initialize plots
    img_display = ax_img.imshow(images_display[0])
    
    # Plot states
    if states is not None:
        state_data = states  # Shape: (T, 5) - [agent_x, agent_y, block_x, block_y, block_theta]
        agent_pos = state_data[:, :2]
        block_pos = state_data[:, 2:4]
        block_theta = state_data[:, 4] if state_data.shape[1] > 4 else None
        
        ax_state.plot(agent_pos[:, 0], agent_pos[:, 1], 'b-', label='Agent', linewidth=2)
        ax_state.plot(block_pos[:, 0], block_pos[:, 1], 'r-', label='Block', linewidth=2)
        ax_state.scatter(agent_pos[0, 0], agent_pos[0, 1], c='blue', s=100, marker='o', zorder=5)
        ax_state.scatter(block_pos[0, 0], block_pos[0, 1], c='red', s=100, marker='s', zorder=5)
        ax_state.scatter(agent_pos[-1, 0], agent_pos[-1, 1], c='blue', s=100, marker='*', zorder=5)
        ax_state.scatter(block_pos[-1, 0], block_pos[-1, 1], c='red', s=100, marker='*', zorder=5)
        ax_state.legend()
        ax_state.set_xlabel('X')
        ax_state.set_ylabel('Y')
        ax_state.grid(True, alpha=0.3)
        ax_state.set_aspect('equal')
    
    # Plot actions
    if actions is not None:
        action_data = actions  # Shape: (T, 2)
        timesteps = np.arange(T)
        ax_action.plot(timesteps, action_data[:, 0], 'g-', label='Action X', linewidth=2)
        ax_action.plot(timesteps, action_data[:, 1], 'm-', label='Action Y', linewidth=2)
        ax_action.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax_action.legend()
        ax_action.grid(True, alpha=0.3)
    
    # Plot contacts
    if n_contacts is not None:
        contact_data = n_contacts.flatten() if len(n_contacts.shape) > 1 else n_contacts
        timesteps = np.arange(T)
        ax_contact.plot(timesteps, contact_data, 'c-', linewidth=2)
        ax_contact.fill_between(timesteps, 0, contact_data, alpha=0.3)
        ax_contact.grid(True, alpha=0.3)
        ax_contact.set_ylim(bottom=0)
    
    # Plot keypoint trajectory
    if keypoints is not None:
        kp_data = keypoints  # Shape: (T, 18, 2) or (T, 36) -> reshape to (T, N, 2)
        if len(kp_data.shape) == 2:
            # Reshape from (T, 36) to (T, 18, 2)
            kp_data = kp_data.reshape(T, -1, 2)
        
        # Plot trajectory of first few keypoints
        for i in range(min(5, kp_data.shape[1])):
            kp_traj = kp_data[:, i, :]
            ax_kp_traj.plot(kp_traj[:, 0], kp_traj[:, 1], '-', alpha=0.6, linewidth=1)
            ax_kp_traj.scatter(kp_traj[0, 0], kp_traj[0, 1], s=50, zorder=5)
        ax_kp_traj.grid(True, alpha=0.3)
    
    # Statistics text
    stats_text = f"Episode {episode_idx}\n"
    stats_text += f"Length: {T} steps\n"
    if states is not None:
        stats_text += f"State shape: {states.shape}\n"
    if keypoints is not None:
        stats_text += f"Keypoints shape: {keypoints.shape}\n"
    if actions is not None:
        stats_text += f"Action shape: {actions.shape}\n"
        stats_text += f"Action range: [{actions.min():.3f}, {actions.max():.3f}]\n"
    if n_contacts is not None:
        stats_text += f"Max contacts: {n_contacts.max()}\n"
    
    ax_stats.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                  family='monospace')
    
    # Animation function
    current_frame = [0]
    
    def update_frame(frame):
        current_frame[0] = frame
        img_display.set_array(images_display[frame])
        ax_img.set_title(f'Episode {episode_idx} - Frame {frame}/{T-1}')
        
        # Update keypoints overlay
        if keypoints is not None:
            ax_kp.clear()
            ax_kp.axis('off')
            ax_kp.set_xlim(0, images_display.shape[2])
            ax_kp.set_ylim(images_display.shape[1], 0)
            
            kp_frame = keypoints[frame]
            if len(kp_frame.shape) == 1:
                kp_frame = kp_frame.reshape(-1, 2)
            
            # Plot keypoints
            ax_kp.scatter(kp_frame[:, 0], kp_frame[:, 1], c='yellow', s=30, 
                         edgecolors='red', linewidths=1, alpha=0.8, zorder=10)
        
        return [img_display]
    
    # Create animation
    anim = animation.FuncAnimation(fig, update_frame, frames=T, interval=100, 
                                   repeat=True, blit=False)
    
    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='ffmpeg', fps=10)
        print("Animation saved!")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize pusht_demo.zarr data')
    parser.add_argument('--path', type=str, default='task_data/pusht_demo.zarr',
                        help='Path to pusht_demo.zarr file')
    parser.add_argument('--episode', type=int, default=0,
                        help='Episode index to visualize (default: 0)')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save animation (e.g., episode_0.mp4). If None, displays interactively')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only print statistics, do not visualize')
    
    args = parser.parse_args()
    
    # Load replay buffer
    data_path = os.path.expanduser(args.path)
    if not os.path.exists(data_path):
        # Try relative to workspace
        workspace_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', args.path)
        if os.path.exists(workspace_path):
            data_path = workspace_path
        else:
            print(f"Error: Cannot find data at {args.path}")
            print(f"Tried: {data_path}")
            print(f"Tried: {workspace_path}")
            return
    
    print(f"Loading data from: {data_path}")
    replay_buffer = ReplayBuffer.create_from_path(data_path, mode='r')
    
    # Print statistics
    print_statistics(replay_buffer)
    
    if not args.stats_only:
        # Visualize episode
        visualize_episode(replay_buffer, args.episode, args.save)


if __name__ == "__main__":
    main()