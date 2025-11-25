"""Test if reward is being collected"""
import numpy as np
import sys
sys.path.append('/media/lht/T7TwoTB/code/fourier_value_functions')

from fvf.env.drone.go_to_target_env import GoToTargetEnv

print("=" * 60)
print("Test Reward Collection")
print("=" * 60)

env = GoToTargetEnv()
obs = env.reset()

print(f"\n[CHECK] Does env have 'reward' attribute?")
print(f"  hasattr(env, 'reward'): {hasattr(env, 'reward')}")

if hasattr(env, 'reward'):
    print(f"  Initial reward list: {env.reward}")
else:
    print(f"  [ERROR] Environment does not have 'reward' attribute!")

print(f"\n[TEST] Run 10 steps and collect rewards:")
total_reward = 0
for step in range(10):
    action = np.random.uniform(-0.1, 0.1, size=3)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    print(f"  Step {step}: reward={reward:.3f}, total={total_reward:.3f}")
    
    if done:
        break

print(f"\n[CHECK] After stepping:")
if hasattr(env, 'reward'):
    print(f"  env.reward list: {env.reward}")
    print(f"  Length: {len(env.reward)}")
    print(f"  Sum: {sum(env.reward):.3f}")
else:
    print(f"  [ERROR] Still no 'reward' attribute!")

print(f"\n[CONCLUSION]")
if hasattr(env, 'reward') and len(env.reward) > 0:
    print(f"  [OK] Reward collection working!")
else:
    print(f"  [ISSUE] Need to add reward collection to environment")