"""
A GoalEnv which wraps the gym Fetch environments

Observation Space: Varies 
Goal Space: Varies
Action Space (3 dim): End-Effector Position Control
"""

from gcsl.envs.goal_env import GoalEnv
from gym.envs.box2d import lunar_lander
from gcsl.envs import lunar_lander_base
from gym import spaces
from collections import OrderedDict
import numpy as np
from multiworld.core.serializable import Serializable
from multiworld.core.serializable import Serializable
from PIL import Image

# state = [
#     (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
#     (pos.y - (self.helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2),
#     vel.x*(VIEWPORT_W/SCALE/2)/FPS,
#     vel.y*(VIEWPORT_H/SCALE/2)/FPS,
#     self.lander.angle,
#     20.0*self.lander.angularVelocity/FPS,
#     1.0 if self.legs[0].ground_contact else 0.0,
#     1.0 if self.legs[1].ground_contact else 0.0
#     ]

# [0, 0, 0, 1, 1]

class LunarEnv(GoalEnv, Serializable):
    def __init__(self, fixed_start=True, fixed_goal=False, frame_skip=2, continuous=False, images=False, image_kwargs=dict()):
        self.frame_skip = frame_skip
        self.quick_init(locals())
        self.inner_env = lunar_lander_base.LunarLander()
        self.inner_env_render = lunar_lander_base.LunarLander()

        self.images = images
        if images:
            state_shape = 3 * 84 * 84 + 8
            self.state_space = spaces.Box(-1, 1, shape=(state_shape,), dtype=np.float32)
            self.observation_space = spaces.Box(-1, 1, shape=(3, 84, 84), dtype=np.float32)
            self.goal_space = spaces.Box(-1, 1, shape=(3, 84, 84), dtype=np.float32)
        else:
            self.state_space = spaces.Box(-1, 1, shape=(8,), dtype=np.float32)
            self.observation_space = spaces.Box(-1, 1, shape=(8,), dtype=np.float32)
            self.goal_space = spaces.Box(-1, 1, shape=(5,), dtype=np.float32)

        self.action_space = self.inner_env.action_space
    
    def render(self, *args, **kwargs):
        return self.inner_env.render(*args, **kwargs)

    def get_image(self, state=None, sample_goal=False):
        if not sample_goal:
            img = self.render('rgb_array')
        else:
            self.inner_env_render.reset(state[:2])
            img = self.inner_env_render.render('rgb_array')

        img = Image.fromarray(img.astype(np.uint8)).resize((84, 84))
        img = np.array(img) / 255.0
        img = np.moveaxis(img, 2, 0)
        return img

    def to_state(self, inner_state, sample_goal=False):
        if not self.images:
            return inner_state
        else:
            im = self.get_image(inner_state, sample_goal)
            im = im.flat[:]
            return np.concatenate([im, inner_state])

    def step(self, action):
        for _ in range(self.frame_skip):
            state, reward, done, info = self.inner_env.step(action)
        return self.to_state(state), 0, False, info

    def from_state(self, state):
        assert self.images
        im_part = state[..., :-8]
        im_part = im_part.reshape(*im_part.shape[:-1], 3, 84, 84)
        state_part = state[..., -8:]
        return im_part, state_part

    def reset(self):
        return self.to_state(self.inner_env.reset(), False)

    def observation(self, state):
        if self.images:
            return self.from_state(state)[0]

        return state
    
    def extract_goal(self, state):
        if self.images:
            return self.from_state(state)[0]

        return state[..., [0,1,4,6,7]]
    
    def _extract_sgoal(self, state):
        if self.images:
            state = self.from_state(state)[1]
        return state[..., [0,1,4,6,7]]

    def goal_distance(self, state, goal_state):
        state_internal = self._extract_sgoal(state)
        goal_state_internal = self._extract_sgoal(goal_state)
        return np.linalg.norm(state_internal[..., [0,1,]] - goal_state_internal[..., [0,1,]], axis=-1)
    
    def euclidean_distance(self, state, goal_state):
        state_internal = self._extract_sgoal(state)
        goal_state_internal = self._extract_sgoal(goal_state)
        return np.linalg.norm(state_internal[..., [0,1,]] - goal_state_internal[..., [0,1,]], axis=-1)
        
    def landed(self, state):
        state_internal = self._extract_sgoal(state)
        return np.all(state_internal[..., -2:], axis=-1)
    
    def sample_goal(self):
        base_goal = np.array([0.0, 0, 0, 0, 0, 0, 1, 1,])
        base_goal[0] += 0.3 * np.random.randn()
        if np.random.rand() > 0.5:
            base_goal[6:] = 0.0
            # base_goal[1] += np.abs(np.random.randn() * 0.3)
            base_goal[2:6] += np.random.randn(4) * 0.2
        return self.to_state(base_goal, True)
        # return base_goal
    
    def get_diagnostics(self, trajectories, desired_goal_states):
        """
        Gets things to log

        Args:
            trajectories: Numpy Array [# Trajectories x Max Path Length x State Dim]
            desired_goal_states: Numpy Array [# Trajectories x State Dim]

        Returns:
            An Ordered Dictionary containing k,v pairs to be logged
        """

        distances = np.array([self.euclidean_distance(trajectories[i], np.tile(desired_goal_states[i], (trajectories.shape[1],1))) for i in range(trajectories.shape[0])])
        landed = np.array([self.landed(trajectories[i]) for i in range(trajectories.shape[0])])
        ypos = np.array([trajectories[i][..., 1] for i in range(trajectories.shape[0])])

        print(np.round([np.min(trajectories[i, :, 1]) for i in range(trajectories.shape[0])], 2))
        print(np.round([np.max(trajectories[i, :, 1]) for i in range(trajectories.shape[0])], 2))
        print(np.round([np.abs(trajectories[i, -1, 1] - desired_goal_states[i, 1]) for i in range(trajectories.shape[0])], 2))
        print(np.round([np.abs(trajectories[i, -1, 0] - desired_goal_states[i, 0]) for i in range(trajectories.shape[0])], 2))

        return OrderedDict([
            ('mean final l2 dist', np.mean(distances[:,-1])),
            ('median final l2 dist', np.median(distances[:,-1])),
            ('mean final landed', np.mean(landed[:,-1])),
            ('mean any landed', np.mean(np.max(landed,axis=1))),
            ('average final y', np.mean(ypos[:, -1])),

        ])