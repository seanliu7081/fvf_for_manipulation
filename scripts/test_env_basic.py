"""Test if environment can produce non-zero rewards"""
import numpy as np
import sys
sys.path.append('/media/lht/T7TwoTB/code/fourier_value_functions')

from fvf.env.drone.go_to_target_env import GoToTargetEnv

print("=" * 60)
print("Test 1: Basic Environment Functionality")
print("=" * 60)

# Create environment
env = GoToTargetEnv()
obs = env.reset()

print(f"\n[SUCCESS] Environment created")
print(f"Observation shape: {obs.shape}")
print(f"Initial position: {obs[:3]}")
print(f"Target position: {env.TARGET_POS}")
print(f"Initial distance: {np.linalg.norm(env.TARGET_POS - obs[:3]):.3f} meters")
print(f"Success threshold: {env.SUCCESS_TH} meters")

print("\n" + "=" * 60)
print("Test 2: Can Random Policy Get Reward?")
print("=" * 60)

episodes_with_reward = 0
max_reward_seen = 0

for episode in range(10):
    obs = env.reset()
    total_reward = 0
    distances = []
    
    for step in range(50):
        # Small random actions
        action = np.random.uniform(-0.1, 0.1, size=3)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        distance = np.linalg.norm(env.TARGET_POS - obs[:3])
        distances.append(distance)
        
        if done:
            break
    
    min_distance = min(distances)
    if total_reward > 0:
        episodes_with_reward += 1
    max_reward_seen = max(max_reward_seen, total_reward)
    
    print(f"Episode {episode}: reward={total_reward:.3f}, min_dist={min_distance:.3f}m")

print(f"\n[STATISTICS]")
print(f"  - Episodes with reward: {episodes_with_reward}/10")
print(f"  - Max reward seen: {max_reward_seen}")
if max_reward_seen > 0:
    print(f"  - Conclusion: [OK] Random policy can get reward")
else:
    print(f"  - Conclusion: [ISSUE] Random policy cannot get reward (sparse reward problem)")

print("\n" + "=" * 60)
print("Test 3: Manual Control to Target")
print("=" * 60)

env = GoToTargetEnv()
obs = env.reset()
current_pos = obs[:3]
target_pos = env.TARGET_POS

print(f"Current position: {current_pos}")
print(f"Target position: {target_pos}")

# Manual control: move towards target each step
total_reward = 0
for step in range(100):
    current_pos = obs[:3]
    direction = target_pos - current_pos
    distance = np.linalg.norm(direction)
    
    # Move towards target (0.05 meters per step)
    if distance > 0.05:
        action = direction / distance * 0.05
    else:
        action = direction
    
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    
    if step % 10 == 0 or done:
        print(f"Step {step}: dist={distance:.3f}m, reward={reward:.3f}, total={total_reward:.3f}")
    
    if done:
        print(f"\n[SUCCESS] Reached target!")
        break

print(f"\n[RESULT] Manual control total reward: {total_reward:.3f}")
if total_reward > 0:
    print(f"Conclusion: [OK] Can get reward by reaching target")
else:
    print(f"Conclusion: [BUG] No reward even when reaching target")