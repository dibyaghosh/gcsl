"""
A GoalEnv which wraps the gym Fetch environments

Observation Space: Varies
Goal Space: Varies
Action Space (3 dim): End-Effector Position Control
"""

from gcsl.envs.goal_env import GoalEnv
from gym import spaces
from collections import OrderedDict
import numpy as np
from multiworld.core.serializable import Serializable
from multiworld.core.serializable import Serializable
import robel
import gym
from gcsl.envs.env_utils import ImageandProprio
class ClawEnv(GoalEnv, Serializable):
    def __init__(self, fixed_start=True, fixed_goal=False, frame_skip=1, continuous=False, images=False, image_kwargs=dict()):
        self.goal = np.array([1., 0.])
        self.frame_skip = frame_skip
        self.quick_init(locals())
        self.inner_env = gym.make("DClawTurnFixed-v0")
        self.images = images
        if not self.images:
            self.state_space = spaces.Box(-np.inf, np.inf, shape=(11,), dtype=np.float32)
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(11,), dtype=np.float32)
            self.goal_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
        else:
            print('Using Images')
            self.state_space = ImageandProprio((3, 84, 84), (11,))
            self.observation_space = ImageandProprio((3, 84, 84), (9,))
            self.goal_space = spaces.Box(0, 1, shape=(3, 84, 84))
            # self.state_space = spaces.Box(-np.inf, np.inf, shape=(11,), dtype=np.float32)
            # self.observation_space = spaces.Box(-np.inf, np.inf, shape=(11,), dtype=np.float32)
            # self.goal_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)

        self.action_space = self.inner_env.action_space

    def render(self, *args, **kwargs):
        return self.inner_env.render(*args, **kwargs)

    def step(self, action):
        for _ in range(self.frame_skip):
            state, reward, done, info = self.inner_env.step(action)
            state = state[:11] # Only first 11 relevant
        if self.images:
            image_obs = self.get_image()
            state = self.state_space.to_flat(image_obs, state)
        return state, 0, False, info

    def reset(self):
        state = self.inner_env.reset()[:11]
        if self.images:
            return self.state_space.to_flat(self.get_image(), state)
        return state

    def observation(self, state):
        if self.images:
            return state[..., :-2]
        else:
            return state

    def extract_goal(self, state):
        if self.images:
            return self.state_space.from_flat(state)[0]
        else:
            return state[..., -2:]

    def _extract_sgoal(self, state):
        return state[..., -2:]

    def get_angles(self, state):
        return np.arctan2(state[..., -1], state[..., -2])

    def goal_distance(self, state, goal_state):
        state_angles = self.get_angles(state)
        goal_angles = self.get_angles(goal_state)
        dist = np.abs(state_angles - goal_angles) % (np.pi * 2)
        return np.minimum(dist, 2 * np.pi - dist)
        # return np.linalg.norm(state[..., [-2, -1]] - goal_state[..., [-2, -1]], axis=-1)

    def euclidean_distance(self, state, goal_state):
        return np.linalg.norm(state[..., [-2, -1]] - goal_state[..., [-2, -1]], axis=-1)

    def get_image(self):
        image_obs = self.inner_env.env.sim.render(84, 84, camera_name='object_target')
        image_obs = np.moveaxis(image_obs, 2, 0) / 255.
        return image_obs

    def set_to_goal(self, goal):
        RESET_POSE = [0, -np.pi / 3, np.pi / 3] * 3

        self.inner_env.env._reset_dclaw_and_object(
            claw_pos=RESET_POSE,
            object_pos=np.arctan2(goal[1], goal[0]),
            object_vel=0,
            guide_pos=self.inner_env.env._target_object_pos)

    def sample_goal(self):
            
        s = self.inner_env.reset()[:11]
        random_angle = np.random.uniform(0, 2 * np.pi)
        goal = np.array([np.cos(random_angle), np.sin(random_angle)])
        s[-2:] = goal
        if self.images:
            self.set_to_goal(goal)
            full_state = self.state_space.to_flat(self.get_image(), s)
            return full_state
        return s

    def get_diagnostics(self, trajectories, desired_goal_states):
        """
        Gets things to log

        Args:
            trajectories: Numpy Array [# Trajectories x Max Path Length x State Dim]
            desired_goal_states: Numpy Array [# Trajectories x State Dim]

        Returns:
            An Ordered Dictionary containing k,v pairs to be logged
        """

        distances = np.array(
            [self.goal_distance(trajectories[i], np.tile(desired_goal_states[i], (trajectories.shape[1], 1))) for i
             in range(trajectories.shape[0])])
        amount_moved = np.array(
            [self.goal_distance(trajectories[i], np.tile(trajectories[i][0], (trajectories.shape[1], 1))) for i
             in range(trajectories.shape[0])])

        return OrderedDict([
            ('mean final angle dist', np.mean(distances[:, -1])),
            ('median final angle dist', np.median(distances[:, -1])),
            ('mean final angle moved', np.mean(amount_moved[:, -1])),
            ('median final angle moved', np.median(amount_moved[:, -1])),

        ])

if __name__ == "__main__":
    import IPython
    IPython.embed()