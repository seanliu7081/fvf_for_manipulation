import numpy as np
import numpy.random as npr
import pybullet as pb
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import (
    DroneModel,
    Physics,
    ActionType,
    ObservationType,
)


class GoToTargetRdEnv(BaseRLAviary):
    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
        gui=False,
        record=False,
    ):
        """Initialization of a single agent environment - RD version.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.workspace = np.array([[-0.75, -0.75, 0], [0.75, 0.75, 0.75]])
        self.EPISODE_LEN_SEC = 8
        self.SUCCESS_TH = 1e-1
        self.seed = None
        self.TARGET_POS = None  # Initialize to avoid None errors
        self.initial_pos = None  # Initialize to avoid None errors
        self.reward = []

        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=ObservationType.KIN,
            act=ActionType.PID,
        )

    def _observationSpace(self):
        obs_lower_bound = np.array([-np.inf, -np.inf, 0, -np.inf, -np.inf, 0, -np.inf, -np.inf, 0])
        obs_upper_bound = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def render2(self):
        return self.render()

    def set_seed(self, seed):
        self.seed = seed

    def reset(self, seed: int = None, options: dict = None):
        seed = self.seed if self.seed else seed
        npr.seed(seed)

        # Randomize target position
        self.TARGET_POS = npr.uniform(self.workspace[0], self.workspace[1])
        self.reward = []
        
        # Randomize initial drone position
        # Keep trying until we get a position that's not too close to target
        min_distance = 0.3  # Minimum distance between start and target
        for _ in range(100):  # Max attempts
            random_init_xyz = npr.uniform(self.workspace[0], self.workspace[1])
            if np.linalg.norm(random_init_xyz - self.TARGET_POS) >= min_distance:
                break
        
        # Set initial position for BaseRLAviary (shape: (1, 3))
        self.INIT_XYZS = random_init_xyz.reshape(1, 3)
        
        # Handle both old and new gym API
        result = super().reset(seed, options)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        
        # Set initial_pos AFTER reset (when drone is positioned)
        drone_obs_full = self._getDroneStateVector(0)
        self.initial_pos = drone_obs_full[:3].copy()
        
        target_idx = pb.createVisualShape(
            pb.GEOM_SPHERE, radius=5e-2, rgbaColor=[1, 0, 0, 1]
        )
        pb.createMultiBody(
            baseVisualShapeIndex=target_idx, basePosition=self.TARGET_POS
        )

        # Get the actual observation using _computeObs to ensure correct format
        obs = self._computeObs()
        
        return obs

    ################################################################################

    def step(self, delta_act):
        """Execute action step"""
        state = self._getDroneStateVector(0)
        pos = state[:3]
        
        # Convert to numpy arrays and flatten
        pos = np.asarray(pos, dtype=np.float32).flatten()
        delta_act = np.asarray(delta_act, dtype=np.float32).flatten()
        
        # Ensure same shape
        if pos.shape[0] != delta_act.shape[0]:
            delta_act = delta_act[:pos.shape[0]]
        
        # Compute action
        act = (pos + delta_act).reshape(1, -1)
        
        # Call parent step
        result = super().step(act)
        
        # Handle both old and new gym API
        if len(result) == 5:
            obs, reward, done, truncated, info = result
        else:
            obs, reward, done, info = result
            truncated = False
        
        # Guard against None observations
        if obs is None:
            obs = self._computeObs()
        
        self.reward.append(reward)
        
        return obs, reward, done, truncated, info

    def _computeReward(self):
        """Computes the current reward value with dense distance-based reward.

        Returns
        -------
        float
            The reward combining distance penalty and success bonus.

        """
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        
        # Calculate distance to target
        distance = np.linalg.norm(self.TARGET_POS - current_pos)
        
        # Dense reward: negative distance (encourages getting closer)
        distance_reward = -distance
        
        # Large bonus for reaching the target
        if distance < self.SUCCESS_TH:
            success_bonus = 10.0
            print(f"[REWARD] SUCCESS! distance={distance:.3f}, reward={distance_reward + success_bonus:.3f}")
        else:
            success_bonus = 0.0
        
        # Combine rewards
        total_reward = distance_reward + success_bonus
        
        return total_reward

    ################################################################################

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        return np.linalg.norm(self.TARGET_POS - state[0:3]) < self.SUCCESS_TH

    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        return False

    def _computeObs(self):
        """Returns the current observation of the environment.
        
        Returns 9 features to match training data structure:
        [0-2]: current drone position
        [3-5]: initial drone position (tracked per episode)
        [6-8]: target position
        """
        try:
            drone_obs_full = self._getDroneStateVector(0)
            drone_current_pos = drone_obs_full[:3]  # Current position
            
            # Track initial position - need to set this in reset()
            if not hasattr(self, 'initial_pos') or self.initial_pos is None:
                self.initial_pos = drone_current_pos.copy()
            
            # Safety check: ensure TARGET_POS is set
            if not hasattr(self, 'TARGET_POS') or self.TARGET_POS is None:
                # Fallback: use zeros if TARGET_POS not set yet
                self.TARGET_POS = np.zeros(3)
            
            obs = np.concatenate([
                drone_current_pos,      # [0-2]: current position
                self.initial_pos,       # [3-5]: initial position (constant per episode)
                self.TARGET_POS         # [6-8]: target position (constant per episode)
            ], axis=-1)
            
            return obs
        except Exception as e:
            # Fallback: return zeros if something goes wrong
            # This prevents None from being returned
            import warnings
            warnings.warn(f"Error in _computeObs: {e}, returning zeros")
            if not hasattr(self, 'initial_pos') or self.initial_pos is None:
                self.initial_pos = np.zeros(3)
            if not hasattr(self, 'TARGET_POS') or self.TARGET_POS is None:
                self.TARGET_POS = np.zeros(3)
            return np.concatenate([
                np.zeros(3),
                self.initial_pos,
                self.TARGET_POS
            ], axis=-1)

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict."""
        state = self._getDroneStateVector(0)
        distance = np.linalg.norm(self.TARGET_POS - state[0:3])
        
        return {
            "is_success": float(distance < self.SUCCESS_TH),
            "distance_to_target": distance
        }