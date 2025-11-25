"""Test which level has the reward attribute"""
import numpy as np
import sys
sys.path.append('/media/lht/T7TwoTB/code/fourier_value_functions')

from fvf.env.drone.go_to_target_env import GoToTargetEnv
from fvf.gym_util.multistep_wrapper import MultiStepWrapper
from fvf.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

print("=" * 60)
print("Test Wrapper Reward Attributes")
print("=" * 60)

# Create wrapped environment (same as in drone_runner)
base_env = GoToTargetEnv()
wrapped_env = MultiStepWrapper(
    VideoRecordingWrapper(
        base_env,
        video_recoder=VideoRecorder.create_h264(
            fps=10,
            codec="h264",
            input_pix_fmt="rgb24",
            crf=22,
            thread_type="FRAME",
            thread_count=1,
        ),
        file_path=None,
    ),
    n_obs_steps=2,
    n_action_steps=1,
    max_episode_steps=100,
)

obs = wrapped_env.reset()

print(f"\n[CHECK] Reward attributes at each level:")
print(f"  MultiStepWrapper has 'reward': {hasattr(wrapped_env, 'reward')}")
print(f"  VideoRecordingWrapper has 'reward': {hasattr(wrapped_env.env, 'reward')}")
print(f"  GoToTargetEnv has 'reward': {hasattr(wrapped_env.env.env, 'reward')}")

print(f"\n[TEST] Run 5 steps:")
for step in range(5):
    action = np.random.uniform(-0.1, 0.1, size=(1, 3))  # (1, 3) for multistep
    obs, reward, done, timeout, info = wrapped_env.step(action)
    print(f"  Step {step}: reward={reward}")

print(f"\n[CHECK] After stepping:")
print(f"  MultiStepWrapper.reward: {getattr(wrapped_env, 'reward', 'NOT FOUND')}")
print(f"  VideoRecordingWrapper.reward: {getattr(wrapped_env.env, 'reward', 'NOT FOUND')}")
print(f"  GoToTargetEnv.reward: {getattr(wrapped_env.env.env, 'reward', 'NOT FOUND')}")

if hasattr(wrapped_env.env.env, 'reward'):
    print(f"\n[OK] Base env has rewards: {len(wrapped_env.env.env.reward)} rewards")
    print(f"     Sample: {wrapped_env.env.env.reward[:3]}")
else:
    print(f"\n[ERROR] Base env has no rewards!")