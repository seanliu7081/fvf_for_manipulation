import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import defaultdict, deque
import dill

def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x,axis=0),n,axis=0)

def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype
    )

def repeated_space(space, n):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f'Unsupported space type {type(space)}')

def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    return np.array(x[-n:])

def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result

# def aggregate(data, method='max'):
#     if method == 'max':
#         # equivalent to any
#         return np.max(data)
#     elif method == 'min':
#         # equivalent to all
#         return np.min(data)
#     elif method == 'mean':
#         return np.mean(data)
#     elif method == 'sum':
#         return np.sum(data)
#     else:
#         raise NotImplementedError()

def aggregate(data, method='max'):
    # Handle empty data
    if len(data) == 0:
        if method in ['max', 'min']:
            return False
        else:
            return 0.0
    
    if method == 'max':
        # equivalent to any
        return np.max(data)
    elif method == 'min':
        # equivalent to all
        return np.min(data)
    elif method == 'mean':
        return np.mean(data)
    elif method == 'sum':
        return np.sum(data)
    else:
        raise NotImplementedError()

# def stack_last_n_obs(all_obs, n_steps):
#     assert(len(all_obs) > 0)
#     all_obs = list(all_obs)
#     result = np.zeros((n_steps,) + all_obs[-1].shape,
#         dtype=all_obs[-1].dtype)
#     start_idx = -min(n_steps, len(all_obs))
#     result[start_idx:] = np.array(all_obs[start_idx:])
#     if n_steps > len(all_obs):
#         # pad
#         result[:start_idx] = result[start_idx]
#     return result

def stack_last_n_obs(all_obs, n_steps):
    assert(len(all_obs) > 0)
    all_obs = list(all_obs)
    
    # Filter out any None values
    all_obs = [obs for obs in all_obs if obs is not None]
    
    # If all observations were None, raise an error
    if len(all_obs) == 0:
        raise ValueError("All observations in stack_last_n_obs are None")
    
    # Double check that the last obs is not None
    if all_obs[-1] is None:
        raise ValueError("Last observation in stack_last_n_obs is None after filtering")
    
    # Check if all_obs[-1] has shape attribute
    if not hasattr(all_obs[-1], 'shape'):
        raise ValueError(f"Last observation does not have shape attribute. Type: {type(all_obs[-1])}, Value: {all_obs[-1]}")
    
    result = np.zeros((n_steps,) + all_obs[-1].shape,
        dtype=all_obs[-1].dtype)
    start_idx = -min(n_steps, len(all_obs))
    result[start_idx:] = np.array(all_obs[start_idx:])
    if n_steps > len(all_obs):
        # pad
        result[:start_idx] = result[start_idx]
    return result


class MultiStepWrapper(gym.Wrapper):
    def __init__(self,
            env,
            n_obs_steps,
            n_action_steps,
            max_episode_steps=None,
            reward_agg_method='max'
        ):
        super().__init__(env)
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.n_obs_steps = n_obs_steps

        self.obs = deque(maxlen=n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.timeout = list()
        self.info = defaultdict(lambda : deque(maxlen=n_obs_steps+1))

    # def reset(self, seed=None, options=None):
    #     """Resets the environment using kwargs."""
    #     obs = super().reset(seed=seed, options=options)

    #     self.obs = deque([obs], maxlen=self.n_obs_steps+1)
    #     self.reward = list()
    #     self.done = list()
    #     self.timeout = list()
    #     self.info = defaultdict(lambda : deque(maxlen=self.n_obs_steps+1))

    #     obs = self._get_obs(self.n_obs_steps)
    #     return obs

    def reset(self, seed=None, options=None):
        """Resets the environment using kwargs."""
        obs = super().reset(seed=seed, options=options)
        
        # Guard against None observation from reset
        if obs is None:
            raise ValueError("Environment reset returned None observation")

        self.obs = deque([obs], maxlen=self.n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.timeout = list()
        self.info = defaultdict(lambda : deque(maxlen=self.n_obs_steps+1))

        obs = self._get_obs(self.n_obs_steps)
        return obs

    # def step(self, action):
    #     """
    #     actions: (n_action_steps,) + action_shape
    #     """
    #     for act in action:
    #         if len(self.done) > 0 and self.done[-1]:
    #             # termination
    #             break
    #         observation, reward, done, timeout, info = super().step(act)

    #         self.obs.append(observation)
    #         self.reward.append(reward)
    #         if (self.max_episode_steps is not None) \
    #             and (len(self.reward) >= self.max_episode_steps):
    #             # truncation
    #             done = True
    #         self.done.append(done)
    #         self.timeout.append(timeout)
    #         self._add_info(info)

    #     observation = self._get_obs(self.n_obs_steps)
    #     reward = aggregate(self.reward, self.reward_agg_method)
    #     done = aggregate(self.done, 'max')
    #     timeout = aggregate(self.timeout, 'max')
    #     info = dict_take_last_n(self.info, self.n_obs_steps)
    #     return observation, reward, done, timeout, info

    def step(self, action):
        """
        actions: (n_action_steps,) + action_shape
        """
        for act in action:
            if len(self.done) > 0 and self.done[-1]:
                # termination
                break
            observation, reward, done, timeout, info = super().step(act)
            
            # Guard against None observations - this can happen when an episode ends
            if observation is None:
                # Use the last valid observation if available, otherwise skip this step
                if len(self.obs) > 0 and self.obs[-1] is not None:
                    observation = self.obs[-1]
                else:
                    # This shouldn't happen, but handle it gracefully by using zeros
                    import warnings
                    warnings.warn("Received None observation and no previous obs available, using zeros")
                    # Get observation shape from the BASE environment's observation space (not wrapper's)
                    if isinstance(self.env.observation_space, spaces.Box):
                        observation = np.zeros(self.env.observation_space.shape, dtype=self.env.observation_space.dtype)
                    else:
                        # For Dict spaces, create a dict of zeros
                        observation = {key: np.zeros(space.shape, dtype=space.dtype) 
                                    for key, space in self.env.observation_space.items()}

            self.obs.append(observation)
            self.reward.append(reward)
            if (self.max_episode_steps is not None) \
                and (len(self.reward) >= self.max_episode_steps):
                # truncation
                done = True
            self.done.append(done)
            self.timeout.append(timeout)
            self._add_info(info)

        observation = self._get_obs(self.n_obs_steps)
        reward = aggregate(self.reward, self.reward_agg_method)
        done = aggregate(self.done, 'max')
        timeout = aggregate(self.timeout, 'max')
        info = dict_take_last_n(self.info, self.n_obs_steps)
        return observation, reward, done, timeout, info

    # def _get_obs(self, n_steps=1):
    #     """
    #     Output (n_steps,) + obs_shape
    #     """
    #     assert(len(self.obs) > 0)
    #     if isinstance(self.observation_space, spaces.Box):
    #         return stack_last_n_obs(self.obs, n_steps)
    #     elif isinstance(self.observation_space, spaces.Dict):
    #         result = dict()
    #         for key in self.observation_space.keys():
    #             result[key] = stack_last_n_obs(
    #                 [obs[key] for obs in self.obs],
    #                 n_steps
    #             )
    #         return result
    #     else:
    #         raise RuntimeError('Unsupported space type')

    def _get_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert(len(self.obs) > 0)
        
        # CRITICAL: Filter out None observations before processing
        # This ensures stack_last_n_obs never receives None
        valid_obs = [obs for obs in self.obs if obs is not None]
        if len(valid_obs) == 0:
            raise ValueError("All observations in _get_obs are None!")
        
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(valid_obs, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs(
                    [obs[key] for obs in valid_obs],
                    n_steps
                )
            return result
        else:
            raise RuntimeError('Unsupported space type')

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)

    def get_rewards(self):
        return self.reward

    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):
        fn = dill.loads(dill_fn)
        return fn(self)

    def get_infos(self):
        result = dict()
        for k, v in self.info.items():
            result[k] = list(v)
        return result
