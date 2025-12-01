"""
Drone image data collection script
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import click
import numpy as np
import pybullet as pb
from fvf.dataset.replay_buffer import ReplayBuffer
from fvf.env.drone.go_to_target_env import GoToTargetEnv


def get_drone_camera_image(env, width=64, height=64):
    """
    Args:
        env: GoToTargetEnv
        width
        height
    
    Returns:
        rgb_image: (H, W, 3) numpy array, uint8
    """

    drone_state = env._getDroneStateVector(0)
    drone_pos = drone_state[:3]
    drone_quat = drone_state[3:7]  # quaternion (x, y, z, w)
    

    rot_matrix = pb.getMatrixFromQuaternion(drone_quat)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    
    forward = rot_matrix[:, 0]

    up = rot_matrix[:, 2]
    
    camera_pos = drone_pos + 0.05 * forward
    
    target_pos = drone_pos + forward
    
    view_matrix = pb.computeViewMatrix(
        cameraEyePosition=camera_pos,
        cameraTargetPosition=target_pos,
        cameraUpVector=up,
        physicsClientId=env.CLIENT
    )
    
    fov = 60
    aspect = width / height
    near = 0.01
    far = 100.0
    projection_matrix = pb.computeProjectionMatrixFOV(
        fov=fov,
        aspect=aspect,
        nearVal=near,
        farVal=far,
        physicsClientId=env.CLIENT
    )

    _, _, rgb, depth, seg = pb.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=pb.ER_TINY_RENDERER,  # CPU renderer, works in DIRECT mode
        physicsClientId=env.CLIENT
    )

    rgb_image = np.array(rgb, dtype=np.uint8)[:, :, :3]
    
    return rgb_image


def get_third_person_camera_image(env, width=128, height=128):
    """
    
    Args:
        env: GoToTargetEnv
        width
        height
    
    Returns:
        rgb_image: (H, W, 3) numpy array, uint8
    """

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
        fov=fov,
        aspect=aspect,
        nearVal=near,
        farVal=far,
        physicsClientId=env.CLIENT
    )

    _, _, rgb, depth, seg = pb.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=pb.ER_TINY_RENDERER,
        physicsClientId=env.CLIENT
    )
    
    rgb_image = np.array(rgb, dtype=np.uint8)[:, :, :3]
    
    return rgb_image


@click.command()
@click.option("-o", "--output", required=True, help="Output zarr path")
@click.option("--num-episodes", default=90, help="Number of successful episodes to collect")
@click.option("--image-width", default=96, help="Image width (default 96 to match PushT)")
@click.option("--image-height", default=96, help="Image height (default 96 to match PushT)")
@click.option("--third-person", is_flag=True, help="Use third-person view instead of drone POV")
@click.option("--both-views", is_flag=True, help="Collect both first-person and third-person views")
def main(output, num_episodes, image_width, image_height, third_person, both_views):
    """
    - img: (T, H, W, 3) uint8
    - keypoint: (T, 3, 3) float32 - 3D keypoints [current_pos, initial_pos, target_pos]
    - action: (T, 3) float32
    - state: (T, 9) float32
    """

    replay_buffer = ReplayBuffer.create_from_path(output, mode="a")
    
    env = GoToTargetEnv(gui=False)
    
    success = 0
    seed = 0
    max_steps = 200
    
    print(f"Collecting {num_episodes} successful episodes...")
    print(f"Image size: {image_width}x{image_height}")
    print(f"View mode: {'third-person' if third_person else 'first-person'}" + 
          (" + third-person" if both_views else ""))
    
    while success < num_episodes:
        episode_data = {
            "img": [],           # 与 PushT 一致
            "keypoint": [],      # 与 PushT 一致
            "action": [],
            "state": [],         # 与 PushT 一致
        }
        if both_views:
            episode_data["img_third_person"] = []
        
        print(f"\nStarting seed {seed}...")
        obs = env.reset(seed=seed)
        done = False
        terminated = False
        
        # Debug info
        print(f"  Target: {env.TARGET_POS.round(3)}")
        print(f"  Initial pos: {obs[:3].round(3)}")
        
        i = 0
        while not done and not terminated and i < max_steps:
            i += 1
            
            # 获取图像
            if third_person:
                image = get_third_person_camera_image(env, image_width, image_height)
            else:
                image = get_drone_camera_image(env, image_width, image_height)
            
            if both_views:
                image_tp = get_third_person_camera_image(env, image_width, image_height)
            
            
            drone_pos = obs[:3]
            initial_pos = obs[3:6]
            target_pos = obs[6:9]
            action = np.clip(target_pos - drone_pos, -0.1, 0.1)
            
            keypoint = np.stack([drone_pos, initial_pos, target_pos], axis=0)
            
            episode_data["img"].append(image)
            episode_data["keypoint"].append(np.float32(keypoint))
            episode_data["action"].append(np.float32(action))
            episode_data["state"].append(np.float32(obs))
            if both_views:
                episode_data["img_third_person"].append(image_tp)
            
            if i % 50 == 0:
                distance = np.linalg.norm(target_pos - drone_pos)
                print(f"  Step {i}: dist={distance:.4f}")
            
            obs, reward, done, terminated, info = env.step(action)
        
        final_distance = np.linalg.norm(env.TARGET_POS - env._getDroneStateVector(0)[:3])
        print(f"  Episode ended: steps={i}, final_dist={final_distance:.4f}")
        
        if final_distance < env.SUCCESS_TH:
            data_dict = {
                "img": np.stack(episode_data["img"]),              # (T, H, W, 3) uint8
                "keypoint": np.stack(episode_data["keypoint"]),    # (T, 3, 3) float32
                "action": np.stack(episode_data["action"]),        # (T, 3) float32
                "state": np.stack(episode_data["state"]),          # (T, 9) float32
            }
            if both_views:
                data_dict["img_third_person"] = np.stack(episode_data["img_third_person"])
            
            replay_buffer.add_episode(data_dict, compressors="disk")
            success += 1
            print(f"  ✓ Saved episode {success}/{num_episodes} (seed {seed})")
        else:
            print(f"  ✗ Failed (seed {seed})")
        
        seed += 1
    
    env.close()
    print(f"\nDone! Collected {success} episodes to {output}")

    print("\n" + "="*60)
    print("DATA SUMMARY (PushT-compatible format)")
    print("="*60)
    print("Expected format (like PushT):")
    print("  img:      (T, H, W, 3) uint8")
    print("  keypoint: (T, N, D) float32  # PushT: (T,18,2), Drone: (T,3,3)")
    print("  action:   (T, A) float32     # PushT: (T,2), Drone: (T,3)")
    print("  state:    (T, S) float32     # PushT: (T,5), Drone: (T,9)")
    print("-" * 60)
    print("Actual collected data:")
    for key in replay_buffer.keys():
        arr = replay_buffer[key]
        print(f"  {key:20s}: shape={str(arr.shape):20s}, dtype={arr.dtype}")


if __name__ == "__main__":
    main()