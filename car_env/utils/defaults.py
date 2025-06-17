import numpy as np
import gymnasium as gym
from dataclasses import dataclass

from car_env.utils.obs_config import ObservationConfig


class DEFAULTS:
    TRACKS = ["icra_2023"]
    OBSERVATION_CONFIG = "obs_config"
    ACTION_SPACE = gym.spaces.Box(
        low=np.array([-1,] * 2),
        high=np.array([1] * 2),
        shape=(2,),
        dtype=np.float32,
    )
    # Constants
    @dataclass
    class SimulatorConstants:
        MIN_VELOCITY = 0.5
        MAX_VELOCITY = 20.8
        PROGRESS_REWARD_SCALE = 1.0
        DEFAULT_STATE_DIMENSION = 12
        MAX_TRACK_SIZE = 175  
        MAX_OMEGA_DOT = 10.0 # [m/s^2] derivitive of wheel speed
        MAX_STEERING = 0.5  # [rad]
        HISTORY_SIZE = 20

        INPUT_REGURGITATION = 0.0001
        INPUT_CHANGE_REGURGITATION = 0.0001

        HEADIND_DIFF_REWARD = 0.01
        CENTERLINE_REWARD = 0.01
        SLIP_REWARD = 0.01
        
        OFF_TRACK_DISTANCE = 0.1  # [m]
        OFF_TRACK_PENALTY = -1.0
        OFF_TRACK_SCALER = -10.0

        SEC_TO_DOUBLE_REWARD = 5.0  # [s]

        FRICTION = 0.8
        MIN_FRICITON = 0.3