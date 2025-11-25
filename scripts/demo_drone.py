import os
import sys

sys.path.insert(0, os.path.abspath("."))

import click
import time
import numpy as np
from fvf.dataset.replay_buffer import ReplayBuffer
from fvf.env.drone.go_to_target_env import GoToTargetEnv
from fvf.env.drone.fly_through_gate_env import FlyThroughGateEnv


@click.command()
@click.option("-o", "--output", required=True)
def main(output):
    # create replay buffer in read-write mode
    replay_buffer = ReplayBuffer.create_from_path(output, mode="a")

    # env = FlyThroughGateEnv(gui=False)
    env = GoToTargetEnv(gui=False)

    success = 0
    seed = 0
    while success < 50:
        episode: list = []
        print(f"starting seed {seed}")

        obs = env.reset(seed=seed)
        done = False
        terminated = False
        # DEBUG: Print the full observation to see what we're getting
        print(f"  Full obs shape: {obs.shape}")
        print(f"  Full obs: {obs}")
        print(f"  env.TARGET_POS: {env.TARGET_POS}")  # The actual target
        print(f"  obs[:3]: {obs[:3]}")
        print(f"  obs[3:6]: {obs[3:6]}")
        print(f"  obs[6:9]: {obs[6:9]}")

        # while not done and not terminated:
        #     i += 1
        #     target_pos = obs[3:].reshape(2, 3).mean(0)
        #     target_pos[2] += 0.1
        #     action = np.clip(target_pos - obs[:3], -0.1, 0.1)
        #     data = {
        #         "obs": np.float32(obs),
        #         "action": np.float32(action),
        #     }
        #     episode.append(data)
        i = 0
        max_steps = 200
        while not done and not terminated and i < max_steps:
            i += 1
            # For GoToTargetEnv: target is directly in obs[3:6]
            drone_pos = obs[:3]
            # target_pos = obs[3:6]
            # target_pos = env.TARGET_POS
            target_pos = obs[6:9]
            
            # Debug: print every 50 steps to see progress
            if i % 50 == 0:
                distance = np.linalg.norm(target_pos - drone_pos)  # target_pos is already obs[6:9]
                print(f"  Step {i}: drone={drone_pos.round(3)}, target={target_pos.round(3)}, dist={distance:.4f}")
            
            action = np.clip(target_pos - drone_pos, -0.1, 0.1)
            data = {
                "obs": np.float32(obs),
                "action": np.float32(action),
            }
            episode.append(data)

            obs, reward, done, terminated, info = env.step(action)

        # After the loop, print why it failed
        print(f"  Episode ended: steps={i}, reward={reward}, done={done}, terminated={terminated}")
        if i >= max_steps:
            print(f"  Failed: Timeout")
        elif terminated:
            print(f"  Failed: Terminated")
        
        final_distance = np.linalg.norm(env.TARGET_POS - env._getDroneStateVector(0)[:3])
        if final_distance < env.SUCCESS_TH:  # Within success threshold

        # if reward == 1:
            data_dict = dict()
            for key in episode[0].keys():
                data_dict[key] = np.stack([x[key] for x in episode])
            replay_buffer.add_episode(data_dict, compressors="disk")
            print(f"saved seed {seed}...")
            success += 1
        else:
            print(f"seed {seed} failed, not saving...")
        seed += 1

    env.close()


if __name__ == "__main__":
    main()
