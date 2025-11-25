"""Check MultiStepWrapper step return"""
import numpy as np
import sys
sys.path.append('/media/lht/T7TwoTB/code/fourier_value_functions')

from fvf.env.drone.go_to_target_env import GoToTargetEnv
from fvf.gym_util.multistep_wrapper import MultiStepWrapper
from fvf.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

print("="*60)
print("Check MultiStepWrapper Return Values")
print("="*60)

# Create wrapped env
base_env = GoToTargetEnv()
wrapped = MultiStepWrapper(
    VideoRecordingWrapper(
        base_env,
        video_recoder=VideoRecorder.create_h264(
            fps=10, codec="h264", input_pix_fmt="rgb24",
            crf=22, thread_type="FRAME", thread_count=1
        ),
        file_path=None
    ),
    n_obs_steps=2,
    n_action_steps=1,
    max_episode_steps=100
)

obs = wrapped.reset()
print(f"\n[RESET] obs shape: {obs.shape}")

# Take a step
action = np.random.uniform(-0.1, 0.1, size=(1, 3))
result = wrapped.step(action)

print(f"\n[STEP] Return values:")
print(f"  Number of returns: {len(result)}")
for i, val in enumerate(result):
    print(f"  Return[{i}]: type={type(val)}, value={val if not isinstance(val, np.ndarray) else f'array shape={val.shape}'}")

if len(result) >= 5:
    obs, reward, done, truncated, info = result
    print(f"\n[PARSED]:")
    print(f"  obs: {type(obs)}, shape={obs.shape if isinstance(obs, np.ndarray) else 'N/A'}")
    print(f"  reward: type={type(reward)}, value={reward}")
    print(f"  done: type={type(done)}, value={done}")
    print(f"  truncated: type={type(truncated)}, value={truncated}")
    print(f"  info: {type(info)}")
    
    print(f"\n[CHECK]:")
    if isinstance(reward, bool) or isinstance(reward, np.bool_):
        print(f"  ❌ ERROR: reward is boolean!")
        print(f"  ❌ This should be a float (the actual reward value)")
    elif isinstance(reward, (int, float, np.number)):
        print(f"  ✅ OK: reward is numeric: {reward}")
    else:
        print(f"  ⚠️  WARNING: unexpected reward type: {type(reward)}")