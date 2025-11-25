"""Check if video recording works"""
import numpy as np
import sys
import pathlib
sys.path.append('/media/lht/T7TwoTB/code/fourier_value_functions')

from fvf.env.drone.go_to_target_env import GoToTargetEnv
from fvf.gym_util.multistep_wrapper import MultiStepWrapper
from fvf.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

print("="*60)
print("Test Video Recording with OpenGL")
print("="*60)

# Create output directory
output_dir = pathlib.Path("test_video_output")
output_dir.mkdir(exist_ok=True)

video_path = output_dir / "test_video_opengl.mp4"
print(f"\n[SETUP] Video path: {video_path}")

# ✅ Create environment with record=True and gui=False
base_env = GoToTargetEnv(gui=False, record=True)
video_wrapper = VideoRecordingWrapper(
    base_env,
    video_recoder=VideoRecorder.create_h264(
        fps=10,
        codec="h264",
        input_pix_fmt="rgb24",
        crf=22,
        thread_type="FRAME",
        thread_count=1,
    ),
    file_path=str(video_path)
)

wrapped = MultiStepWrapper(
    video_wrapper,
    n_obs_steps=2,
    n_action_steps=1,
    max_episode_steps=50
)

print(f"\n[CHECK] video_wrapper.file_path: {video_wrapper.file_path}")
print(f"[CHECK] Has video_recoder: {hasattr(video_wrapper, 'video_recoder')}")

# Start recording
print(f"\n[START] Starting video recording...")
video_wrapper.video_recoder.start(str(video_path))

# Run episode
obs = wrapped.reset()
for i in range(50):
    action = np.random.uniform(-0.1, 0.1, size=(1, 3))
    obs, reward, done, truncated, info = wrapped.step(action)
    
    # Call render to generate frames
    frame = base_env.render()  # ✅ Important!
    
    if done or truncated:
        break

# Stop recording
print(f"\n[STOP] Stopping video recording...")
video_wrapper.video_recoder.stop()

# Check if file exists
import os
if os.path.exists(video_path):
    size = os.path.getsize(video_path)
    print(f"\n[SUCCESS] ✅ Video saved: {video_path}")
    print(f"[SUCCESS] File size: {size / 1024:.2f} KB")
else:
    print(f"\n[FAILED] ❌ Video not created: {video_path}")