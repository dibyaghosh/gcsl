"""Helper functions to create envs easily through one interface.

- create_env(env_name)
- get_goal_threshold(env_name)
"""

import numpy as np
from gcsl.envs.env_utils import DiscretizedActionEnv

try:
    import mujoco_py
except:
    print('MuJoCo must be installed.')

from gcsl.envs.room_env import PointmassGoalEnv
from gcsl.envs.sawyer_push import SawyerPushGoalEnv
from gcsl.envs.sawyer_door import SawyerDoorGoalEnv
from gcsl.envs.lunarlander import LunarEnv
from gcsl.envs.claw_env import ClawEnv

env_names = ['pointmass_rooms', 'pointmass_empty', 'pusher', 'lunar', 'door', 'claw']

def create_env(env_name):
    """Helper function."""
    assert env_name in env_names
    if env_name == 'pusher':
        return SawyerPushGoalEnv()
    elif env_name == 'door':
        return SawyerDoorGoalEnv()
    elif env_name == 'pointmass_empty':
        return PointmassGoalEnv(room_type='empty')
    elif env_name == 'pointmass_rooms':
        return PointmassGoalEnv(room_type='rooms')
    elif env_name == 'lunar':
        return LunarEnv()
    elif env_name == 'claw':
        return ClawEnv()

def get_env_params(env_name, images=False):
    assert env_name in env_names

    base_params = dict(
        eval_freq=10000,
        eval_episodes=50,
        max_trajectory_length=50,
        max_timesteps=1e6,
    )

    if env_name == 'pusher':
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    elif 'pick' in env_name:
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    elif env_name == 'door':
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    elif 'pointmass' in env_name:
        env_specific_params = dict(
            goal_threshold=0.08,
            max_timesteps=2e5,
            eval_freq=2000,
        )
    elif env_name == 'lunar':
        env_specific_params = dict(
            goal_threshold=0.08,
            max_timesteps=2e5,
            eval_freq=2000,
        )

    elif env_name == 'claw':
        env_specific_params = dict(
            goal_threshold=0.1,
        )
    else:
        raise NotImplementedError()
    
    base_params.update(env_specific_params)
    return base_params