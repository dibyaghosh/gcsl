import numpy as np
import room_world.rooms
from room_world.room_env import RoomEnv
from multiworld.core.serializable import Serializable
from collections import OrderedDict, Sequence
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict, get_asset_full_path

def pointmass_camera_config(cam):
    cam.lookat[:3] = np.array([0,0,0])
    cam.distance = 1.6
    cam.azimuth = -90
    cam.elevation = -90.0
    
class PMEnv(RoomEnv):   

    FRAME_SKIP = 5
    MAX_PATH_LENGTH = 200

    def __init__(self,
                 room=None, # Specify either room or room type
                 room_type='empty', # Choose from ['empty', 'wall', 'rooms']
                 potential_type="euclidean", # Choose from ['none' (no shaping) ,'shaped' (shortest distance between COMs), 'euclidean' (euclidean distance between states)]
                 base_reward='positive', # Choose from ['positive'(0,1), 'negative' (-1,0)]
                 shaped=False,
                 speed=1,
                 *args, **kwargs
                ):
        
        Serializable.quick_init(self, locals())
        self.use_images = False
        room_defaults = dict(
            empty=room_world.rooms.Room('pm', 1.2, 1.2), 
            wall=room_world.rooms.RoomWithWall('pm', 1.2, 1.2),
            rooms=room_world.rooms.FourRoom('pm', 1.2, 1.2),
        )
        if room is None:
            room = room_defaults[room_type]

        super().__init__(
            room=room,
            potential_type=potential_type,
            shaped=shaped,
            base_reward=base_reward,
            *args, **kwargs
        )

        new_frame_skip = int(np.ceil(self.frame_skip * speed))
        self.modifier = self.frame_skip * speed / new_frame_skip
        self.frame_skip = new_frame_skip

    def preprocess(self, action):
        return action * self.modifier

    def _get_env_obs(self):
        return self.get_body_com("particle")[:2].copy()

    def _get_env_achieved_goal(self, obs):
        return obs

    def viewer_setup(self):
        pointmass_camera_config(self.viewer.cam)

    def sample_goal_joints(self):
        return np.zeros((0,))

    def get_potential(self, achieved_goal, desired_goal):
        if self.potential_type == 'shaped':
            return -1 * self.room.get_shaped_distance(achieved_goal, desired_goal)
        elif self.potential_type == 'euclidean':
            return -1 * np.linalg.norm(achieved_goal-desired_goal)
        elif self.potential_type == 'none':
            return 0
        else:
            raise NotImplementedError()

    def get_base_reward(self, achieved_goal, desired_goal):
        euclidean_dist = np.linalg.norm(achieved_goal - desired_goal)
        base_reward_modifier = 1 if self.base_reward == 'positive' else 0
        if euclidean_dist < 0.05:
            return 0 + base_reward_modifier
        return -1 + base_reward_modifier
    
    def _get_info(self, obs):
        current_state = obs['state_achieved_goal']

        return dict(
            euclidean_distance=np.linalg.norm(current_state-self.goal),
            shaped_distance=self.room.get_shaped_distance(current_state, self.goal),
            position=current_state,
        )

    def _reset_to_xy(self, pos):
        
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        qpos[0:2] = pos - self.baseline_start
        qvel[0:2] = 0

        self.set_state(qpos, qvel)

    def set_to_goal(self, goal):
        self._reset_to_xy(goal['state_desired_goal'][:2])

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'euclidean_distance',
            'shaped_distance',
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'Final %s%s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
            ))
        return statistics

    def sample_goals(self, batch_size):
        goals = np.zeros((batch_size, 2))
        goals[:] = self.possible_positions[np.random.choice(len(self.possible_positions), batch_size, replace=True)]
        transformed_goals = self._get_obs_goals(goals)

        return {
            'desired_goal': transformed_goals,
            'state_desired_goal': goals,
        }