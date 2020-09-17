import numpy as np

import gym
from gym.spaces import Dict, Box, Discrete

from multiworld.core.serializable import Serializable
from multiworld.core.wrapper_env import ProxyEnv

class DiscreteActionMultiWorldEnv(ProxyEnv, Serializable):
    def __init__(self, wrapped_env, granularity=3):
        self.quick_init(locals())
        ProxyEnv.__init__(self, wrapped_env)
        actions_meshed = np.meshgrid(*[np.linspace(lo, hi, granularity) for lo,hi in zip(self.wrapped_env.action_space.low, self.wrapped_env.action_space.high)])
        self.base_actions = np.array([a.flat[:] for a in actions_meshed]).T
        self.action_space = Discrete(len(self.base_actions))

    def step(self, action):
        return self.wrapped_env.step(self.base_actions[action])

class MultiWorldEnvWrapper(gym.Env, Serializable):
    """
        MULTIWORLD -> GYM Wrapper
    """
    def __init__(
            self,
            env,
            obs_keys=None,
            goal_keys=None,
            append_goal_to_obs=True,
    ):
        Serializable.quick_init(self, locals())
        self.env = env

        if obs_keys is None:
            obs_keys = ['observation']
        if goal_keys is None:
            goal_keys = ['desired_goal']
        if append_goal_to_obs:
            obs_keys += goal_keys
        for k in obs_keys:
            assert k in self.env.observation_space.spaces
        
        assert isinstance(self.env.observation_space, Dict)

        self.obs_keys = obs_keys
        self.goal_keys = goal_keys
        # TODO: handle nested dict
        self.observation_space = Box(
            np.hstack([
                self.env.observation_space.spaces[k].low
                for k in obs_keys
            ]),
            np.hstack([
                self.env.observation_space.spaces[k].high
                for k in obs_keys
            ]),
        )
        self.action_space = self.env.action_space


    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        flat_obs = np.hstack([obs[k] for k in self.obs_keys])
        return flat_obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        return np.hstack([obs[k] for k in self.obs_keys])
    
    def render(self, **kwargs):
        return self.env.render(**kwargs)
    
    def log_diagnostics(self, paths, logger=None,**kwargs):
        stats = self.env.get_diagnostics(paths)
        for k, v in stats.items():
            if logger:
                logger.record_tabular(k, v)
            else:
                print('{0:<20} {1:<10}'.format(k,v))

    @property
    def wrapped_env(self):
        return self.env

    def __getattr__(self, attrname):
        if attrname == '_serializable_initialized':
            return None
        return getattr(self.env, attrname)

    def get_param_values(self):
        if hasattr(self.env, 'get_param_values'):
            return self.env.get_param_values()
        else:
            return dict()

    def set_param_values(self, params):
        if hasattr(self.env, 'set_param_values'):
            return self.env.set_param_values(params)
        else:
            return
