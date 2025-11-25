"""PyBullet video recording without GUI"""
import gymnasium as gym
import numpy as np
import pybullet as p
import imageio
import os

class PyBulletVideoWrapper(gym.Wrapper):
    def __init__(self, env, video_path=None, fps=10, camera_distance=1.5, width=640, height=480):
        super().__init__(env)
        self.video_path = video_path
        self.fps = fps
        self.camera_distance = camera_distance
        self.width = width
        self.height = height
        self.frames = []
        self.is_recording = video_path is not None
        
    def reset(self, **kwargs):
        self.frames = []
        return self.env.reset(**kwargs)
    
    def step(self, action):
        # Ensure action is a numpy array with correct shape
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        
        obs, reward, done, truncated, info = self.env.step(action)
        
        if self.is_recording:
            frame = self._capture_frame()
            if frame is not None:
                self.frames.append(frame)
        
        return obs, reward, done, truncated, info
    
    def _capture_frame(self):
        """Capture frame using PyBullet's getCameraImage"""
        try:
            # Get drone position
            if hasattr(self.env, 'DRONE_IDS'):
                drone_id = self.env.DRONE_IDS[0]
                drone_pos, _ = p.getBasePositionAndOrientation(drone_id)
            else:
                drone_pos = [0, 0, 0.5]
            
            # Camera setup
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=drone_pos,
                distance=self.camera_distance,
                yaw=45,
                pitch=-30,
                roll=0,
                upAxisIndex=2
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=self.width / self.height,
                nearVal=0.1,
                farVal=100.0
            )
            
            # Capture
            _, _, rgb, _, _ = p.getCameraImage(
                width=self.width,
                height=self.height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            
            return np.array(rgb)[:, :, :3]
            
        except Exception as e:
            return None
    
    def save_video(self):
        """Save accumulated frames to video"""
        if self.is_recording and len(self.frames) > 0:
            try:
                print(f"[VIDEO] Saving {len(self.frames)} frames to {self.video_path}")
                imageio.mimwrite(self.video_path, self.frames, fps=self.fps, quality=8)
                
                # Verify file was created
                if os.path.exists(self.video_path):
                    file_size = os.path.getsize(self.video_path) / 1024
                    print(f"[VIDEO] Saved: {self.video_path} ({file_size:.2f} KB)")
                    return True
                else:
                    print(f"[VIDEO] File not created: {self.video_path}")
                    return False
            except Exception as e:
                print(f"[VIDEO] Failed to save: {e}")
                return False
        return False
    
    def close(self):
        self.save_video()
        return self.env.close()