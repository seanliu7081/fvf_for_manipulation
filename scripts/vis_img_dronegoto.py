"""
Visualize drone image data collected with collect_drone_image_data.py
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import click
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from fvf.dataset.replay_buffer import ReplayBuffer


def visualize_episode(replay_buffer, episode_idx, save_path=None):
    """
    可视化单个 episode
    
    显示：
    - 图像序列（采样几帧）
    - 3D 轨迹图
    - keypoint 信息
    """
    episode = replay_buffer.get_episode(episode_idx)
    
    img = episode["img"]           # (T, H, W, 3)
    keypoint = episode["keypoint"] # (T, 3, 3) - [current, initial, target]
    action = episode["action"]     # (T, 3)
    state = episode["state"]       # (T, 9)
    
    T = len(img)
    print(f"\nEpisode {episode_idx}: {T} timesteps")
    print(f"  img shape: {img.shape}")
    print(f"  keypoint shape: {keypoint.shape}")
    print(f"  action shape: {action.shape}")
    
    # 提取轨迹
    current_pos = keypoint[:, 0, :]   # (T, 3) - 无人机位置
    initial_pos = keypoint[0, 1, :]   # (3,) - 初始位置（固定）
    target_pos = keypoint[0, 2, :]    # (3,) - 目标位置（固定）
    
    print(f"  Initial pos: {initial_pos.round(3)}")
    print(f"  Target pos: {target_pos.round(3)}")
    print(f"  Final pos: {current_pos[-1].round(3)}")
    print(f"  Final distance: {np.linalg.norm(target_pos - current_pos[-1]):.4f}")
    
    # 创建图形
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 5, figure=fig, hspace=0.3, wspace=0.3)
    
    # 上面两行：显示 8 帧图像
    num_frames = 8
    frame_indices = np.linspace(0, T-1, num_frames, dtype=int)
    
    for i, idx in enumerate(frame_indices):
        row = i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img[idx])
        ax.set_title(f"t={idx}", fontsize=10)
        ax.axis("off")
        
        # 在图像上标注位置信息
        pos = current_pos[idx]
        ax.text(5, 10, f"pos: {pos.round(2)}", fontsize=7, color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    # 右侧：3D 轨迹图
    ax3d = fig.add_subplot(gs[:2, 4], projection='3d')
    
    # 绘制轨迹
    ax3d.plot(current_pos[:, 0], current_pos[:, 1], current_pos[:, 2], 
              'b-', linewidth=2, label='Trajectory')
    
    # 标记起点、终点、目标
    ax3d.scatter(*initial_pos, c='green', s=100, marker='o', label='Start')
    ax3d.scatter(*current_pos[-1], c='blue', s=100, marker='s', label='End')
    ax3d.scatter(*target_pos, c='red', s=100, marker='*', label='Target')
    
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_title('3D Trajectory')
    ax3d.legend(loc='upper left', fontsize=8)
    
    # 设置相同的轴范围
    all_pos = np.vstack([current_pos, initial_pos.reshape(1, 3), target_pos.reshape(1, 3)])
    max_range = np.max(all_pos.max(axis=0) - all_pos.min(axis=0)) / 2
    mid = all_pos.mean(axis=0)
    ax3d.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax3d.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax3d.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    # 底部：动作和距离曲线
    ax_action = fig.add_subplot(gs[2, :2])
    ax_action.plot(action[:, 0], label='action_x', alpha=0.8)
    ax_action.plot(action[:, 1], label='action_y', alpha=0.8)
    ax_action.plot(action[:, 2], label='action_z', alpha=0.8)
    ax_action.set_xlabel('Timestep')
    ax_action.set_ylabel('Action')
    ax_action.set_title('Actions over time')
    ax_action.legend(loc='upper right', fontsize=8)
    ax_action.grid(True, alpha=0.3)
    
    ax_dist = fig.add_subplot(gs[2, 2:4])
    distances = np.linalg.norm(target_pos - current_pos, axis=1)
    ax_dist.plot(distances, 'r-', linewidth=2)
    ax_dist.axhline(y=0.1, color='g', linestyle='--', label='Success threshold')
    ax_dist.set_xlabel('Timestep')
    ax_dist.set_ylabel('Distance to target')
    ax_dist.set_title('Distance to target over time')
    ax_dist.legend(loc='upper right', fontsize=8)
    ax_dist.grid(True, alpha=0.3)
    
    # 统计信息
    ax_info = fig.add_subplot(gs[2, 4])
    ax_info.axis('off')
    info_text = f"""Episode {episode_idx}
    
Timesteps: {T}
Initial: {initial_pos.round(3)}
Target: {target_pos.round(3)}
Final: {current_pos[-1].round(3)}

Final dist: {distances[-1]:.4f}
Min dist: {distances.min():.4f}
Success: {distances[-1] < 0.1}
"""
    ax_info.text(0.1, 0.9, info_text, transform=ax_info.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Drone GoTo Episode {episode_idx}', fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to {save_path}")
    
    plt.show()
    plt.close()


def create_video(replay_buffer, episode_idx, output_path):
    """
    创建 episode 的视频
    """
    try:
        import imageio
    except ImportError:
        print("Please install imageio: pip install imageio imageio-ffmpeg")
        return
    
    import io
    from PIL import Image
    
    episode = replay_buffer.get_episode(episode_idx)
    img = episode["img"]  # (T, H, W, 3)
    keypoint = episode["keypoint"]
    
    current_pos = keypoint[:, 0, :]
    target_pos = keypoint[0, 2, :]
    
    frames = []
    print(f"Creating video with {len(img)} frames...")
    
    for t in range(len(img)):
        # 创建带标注的帧
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.imshow(img[t])
        
        pos = current_pos[t]
        dist = np.linalg.norm(target_pos - pos)
        
        ax.set_title(f"t={t}, dist={dist:.3f}", fontsize=10)
        ax.axis("off")
        
        # 使用 buffer 保存图像，兼容所有 backend
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        frame = np.array(Image.open(buf))[:, :, :3]  # 去掉 alpha 通道
        frames.append(frame)
        buf.close()
        plt.close(fig)
        
        if (t + 1) % 20 == 0:
            print(f"  Processed {t + 1}/{len(img)} frames")
    
    # 保存视频
    imageio.mimsave(output_path, frames, fps=10)
    print(f"Video saved to {output_path}")


@click.command()
@click.option("-i", "--input", "input_path", required=True, help="Input zarr path")
@click.option("-e", "--episode", default=0, help="Episode index to visualize")
@click.option("--all", "show_all", is_flag=True, help="Show all episodes (one by one)")
@click.option("--save-dir", default=None, help="Directory to save visualizations")
@click.option("--video", is_flag=True, help="Create video for the episode")
def main(input_path, episode, show_all, save_dir, video):
    """
    Visualize drone image data
    """
    print(f"Loading data from: {input_path}")
    replay_buffer = ReplayBuffer.copy_from_path(input_path)
    
    n_episodes = replay_buffer.n_episodes
    n_steps = replay_buffer.n_steps
    
    print("=" * 60)
    print("REPLAY BUFFER STRUCTURE")
    print("=" * 60)
    print(f"Total episodes: {n_episodes}")
    print(f"Total timesteps: {n_steps}")
    print(f"Episode lengths: {replay_buffer.episode_lengths}")
    print("-" * 60)
    print("Data keys:")
    for key in replay_buffer.keys():
        arr = replay_buffer[key]
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
    print("=" * 60)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    if show_all:
        for ep_idx in range(n_episodes):
            save_path = os.path.join(save_dir, f"episode_{ep_idx:03d}.png") if save_dir else None
            visualize_episode(replay_buffer, ep_idx, save_path)
    else:
        save_path = os.path.join(save_dir, f"episode_{episode:03d}.png") if save_dir else None
        visualize_episode(replay_buffer, episode, save_path)
        
        if video:
            video_path = os.path.join(save_dir or ".", f"episode_{episode:03d}.mp4")
            create_video(replay_buffer, episode, video_path)


if __name__ == "__main__":
    main()