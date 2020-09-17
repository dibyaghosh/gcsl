# Base imports
import numpy as np

# Multiworld Imports
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.core.serializable import Serializable
from multiworld.envs.mujoco.mujoco_env import MujocoEnv
import mujoco_py

# Gym / rllab imports
import gym
from gym.spaces import Dict, Box
from gym.utils import EzPickle


# Misc imports
from collections import OrderedDict, Sequence


class RoomEnv(MujocoEnv, MultitaskEnv, Serializable):

    FRAME_SKIP = 5
    MAX_PATH_LENGTH = 200

    def __init__(self,
                # Room
                room,
                # Start and Goal
                 start_config="all",
                 goal_config="all",
                # Reward
                 potential_type='',
                 shaped=False,
                 base_reward='',
                # State and Goal Representations
                 use_state_images=False,
                 use_goal_images=False,
                 image_resolution=64,
                 # Time Limits 
                 max_path_length=None,
                 *args, **kwargs
                ):

        # Initialize
        Serializable.quick_init(self, locals())
        MultitaskEnv.__init__(self)
        
        
        # Environment Configuration
        self._room = room
        model = self.room.get_mjcmodel()
        self.possible_positions = self.room.XY(n=50)

        with model.asfile() as f:
            MujocoEnv.__init__(self, f.name, frame_skip=self.FRAME_SKIP)


        # Initialization 
        self.start_config = start_config
        self.baseline_start = self.room.get_start()
        self.start = np.zeros_like(self.baseline_start)

        self.goal_config = goal_config
        self.baseline_goal = self.room.get_target()
        self.goal = np.zeros_like(self.baseline_goal)

        # Time Limit
        self.curr_path_length = 0
        if max_path_length is None:
            self.max_path_length = self.MAX_PATH_LENGTH
        else:
            self.max_path_length = max_path_length

        # Reward Functions
        self.potential_type = potential_type
        self.shaped = shaped
        self.base_reward = base_reward

        # Action Space
        bounds = self.model.actuator_ctrlrange.copy()
        self.action_space = Box(low=bounds[:, 0], high=bounds[:, 1])

        self.use_state_images = use_state_images
        self.use_goal_images = use_goal_images
        self.image_resolution = image_resolution

        # Observation Space
        example_state_obs = self._get_env_obs()
        if self.use_state_images:
            example_obs = self.get_image(self.image_resolution, self.image_resolution, camera_name='topview')
        else:
            example_obs = example_state_obs
        
        state_obs_shape = example_obs.shape
        obs_shape = example_obs.shape
        self.obs_space = Box(-1 * np.ones(obs_shape),np.ones(obs_shape))
        self.state_obs_space = Box(-1 * np.ones(state_obs_shape), np.ones(state_obs_shape))


        # Goal Space
        
        
        example_state_goal = self._get_env_achieved_goal(example_state_obs)
        if self.use_goal_images:
            example_goal = self.get_image(self.image_resolution, self.image_resolution, camera_name='topview')
        else:
            example_goal = example_state_goal

        state_goal_shape = example_state_goal.shape
        goal_shape = example_goal.shape
        self.goal_space = Box(-1 * np.ones(goal_shape), np.ones(goal_shape))
        self.state_goal_space =  Box(-1 * np.ones(state_goal_shape), np.ones(state_goal_shape))

        # Final Setup
        self.observation_space = Dict([
            ('observation', self.obs_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.goal_space),
            ('state_observation', self.state_obs_space),
            ('state_desired_goal', self.state_goal_space),
            ('state_achieved_goal', self.state_goal_space),
        ])

        self.reset()

    @property
    def room(self):
        return self._room

    def _get_env_obs(self) :
        raise NotImplementedError()
    
    def _get_env_achieved_goal(self, obs):
        raise NotImplementedError()

    def viewer_setup(self):
        raise NotImplementedError()

    def sample_point(self, config):
        if config == 'all':
            i = np.random.choice(self.possible_positions.shape[0])
            return self.possible_positions[i]
        else:
            raise NotImplementedError()

    def sample_point_range(self, lb, ub):
        lb = np.array(lb)
        ub = np.array(ub)
        return np.random.rand(*lb.shape) * (ub - lb) + lb
    
    def sample(self, config, baseline):
        if isinstance(config, str):
            return self.sample_point(config)
        elif isinstance(config, Sequence):
            return self.sample_point_range(*config)
        else:
            return baseline
        
    def sample_start_position(self):
        sample = self.sample(self.start_config, self.baseline_start.copy())
        return sample
    
    def sample_goal_position(self):
        return self.sample(self.goal_config, self.baseline_goal.copy())
    
    def sample_goal_joints(self):
        raise NotImplementedError()

    def get_potential(self, achieved_goal, desired_goal):
        raise NotImplementedError()
    
    def get_base_reward(self, achieved_goal, desired_goal):
        raise NotImplementedError()

    def get_reward(self, achieved_goal, desired_goal):
        
        base_reward = self.get_base_reward(achieved_goal, desired_goal)
        potential = self.get_potential(achieved_goal, desired_goal)

        if self.shaped:
            reward = base_reward + (potential - self.previous_potential)
            self.previous_potential = potential
        else:
            reward = potential

        return reward

    def compute_rewards(self, actions, obs):
        if len(obs['achieved_goal'][0]) != 2:
            raise NotImplementedError() # Todo: Handle Images
        achieved_goals = np.array(obs['achieved_goal'])
        desired_goals = np.array(obs['desired_goal'])
        rewards = np.array([
            self.get_reward(state, goal)
            for state, goal in zip(achieved_goals, desired_goals)
        ])
        return rewards

    def _get_obs(self):
        state_obs = self._get_env_obs()
        achieved_state_goal = self._get_env_achieved_goal(state_obs).copy()
        intended_state_goal = self.goal.copy()

        if self.use_state_images:
            obs = self.get_image(self.image_resolution, self.image_resolution, 'topview').copy()
        else:
            obs = state_obs.copy()
        
        if self.use_goal_images:
            achieved_goal = self.get_image(self.image_resolution, self.image_resolution, 'topview').copy()
            intended_goal = self.goal_image.copy()
        else:
            achieved_goal = achieved_state_goal.copy()
            intended_goal = intended_state_goal.copy()
            
        return dict(
            observation=obs,
            desired_goal=intended_goal,
            achieved_goal=achieved_goal,
            state_observation=state_obs,
            state_desired_goal=intended_state_goal,
            state_achieved_goal=achieved_state_goal,
        )
    
    def preprocess(self, action):
        return action
    
    def step(self, action):
        
        action = self.preprocess(action)
        self.do_simulation(action, self.frame_skip)

        self._set_goal_marker(self.goal_position[:2]) # Need to do this every turn to prevent it from being moved for some reason

        obs = self._get_obs()

        reward = self.get_reward(obs['state_achieved_goal'], obs['state_desired_goal'])
        info = self._get_info(obs)
        self.curr_path_length += 1
        done = self.curr_path_length >= self.max_path_length
        return obs, reward, done, info

    def _set_goal_marker(self, goal):
        self.data.site_xpos[self.model.site_name2id('goal')] = [*goal, 0]

    def _reset_to_xy(self, pos):
        raise NotImplementedError()


    def reset_model(self):


        # Start
        self.start_position = self.sample_start_position()
        self._reset_to_xy(self.start_position)

        # Goal
        self.goal_position = self.sample_goal_position()
        self.goal_joints = self.sample_goal_joints()
        self.goal = np.concatenate([self.goal_position, self.goal_joints])
        if self.use_goal_images:
            self.goal_image = self._get_goal_images([self.goal])[0]
        
        self.curr_path_length = 0
        obs = self._get_obs()
        self.previous_potential = self.get_potential(obs['state_achieved_goal'], obs['state_desired_goal'])

        return obs

    @property
    def goal_dim(self) -> int:
        return np.prod(self.goal_space.shape)

    def _get_goal_images(self, state_goals):
        current_state = self.sim.get_state()
        qvel = self.init_qvel.copy()
        images = []
        for goal in state_goals:
            self.set_to_goal(dict(state_desired_goal=goal))
            images.append(self.sim.render(self.image_resolution, self.image_resolution, camera_name='topview'))

        self.sim.set_state(current_state)
        self.sim.forward()
        return np.array(images)
    
    def _get_obs_goals(self, state_goals):
        if self.use_goal_images:
            return self._get_goal_images(state_goals)
        else:
            return state_goals.copy()
    
    def get_goal(self):
        state_desired_goal = self.goal
        desired_goal = self._get_obs_goals([state_desired_goal])[0]

        return {
            'desired_goal': desired_goal,
            'state_desired_goal': self.goal,
        }

    def sample_goals(self, batch_size):
        # Consider reimplementing this function to make it much faster

        goals = np.zeros((batch_size, self.state_goal_space.shape[0]))
        for i in range(batch_size):
            goals[i,:2] = self.sample_goal_position()
            goals[i, 2:] = self.sample_goal_joints()
        
        transformed_goals = self._get_obs_goals(goals)

        return {
            'desired_goal': transformed_goals,
            'state_desired_goal': goals,
        }
    

    def _get_info(self,obs):
        raise NotImplementedError()
        
    def get_diagnostics(self, paths, prefix=''):
        raise NotImplementedError()

    def set_to_goal(self, goal):
        raise NotImplementedError()

    def get_env_state(self):
        joint_state = self.sim.get_state()
        goal = self.goal.copy()
        return joint_state, goal

    def set_env_state(self, state):
        state, goal = state
        self.sim.set_state(state)
        self.sim.forward()
        self.goal = goal