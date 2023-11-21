from __future__ import annotations

import os
import json
import numpy as np
from omegaconf import DictConfig
import gymnasium as gym
import logging
import carla
from omegaconf import DictConfig, OmegaConf
from ray.rllib.env.env_context import EnvContext
from typing import Optional

from environment.sensor_interface import SensorInterface
from srunner.tools.route_manipulation import downsample_route
from configs import g_conf, merge_with_yaml, set_type_of_process
from dataloaders.transforms import encode_directions_4, encode_directions_6

from .waypointer import Waypointer
from environment import Environment
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from agents.navigation.local_planner import RoadOption

logger = logging.getLogger(__name__)

def checkpoint_parse_configuration_file(filename):
    with open(filename, 'r') as f:
        configuration_dict = json.loads(f.read())

    return configuration_dict['yaml'], configuration_dict['checkpoint'], \
           configuration_dict['agent_name']

class CILv2_env(gym.Env):
    def __init__(self,
                 env_config: DictConfig | dict,
                 path_to_conf_file: str,
                 rllib_config: Optional[EnvContext] = None,
                 ):
        # for the RLlib training:
        # update the port so every worker has different port
        if rllib_config:
            offset = rllib_config.worker_index
            port = env_config.get('port', 2000) + 2*offset
            tm_port = env_config.get('tm_port', 8000) + offset
            env_config.update({'port': port, 'tm_port': tm_port})

        if not isinstance(env_config, DictConfig):
            env_config = OmegaConf.create(dict(env_config))

        #  current global plans to reach a destination
        self._global_plan = None
        self._global_plan_world_coord = None

        self.left_change_count = 0
        self.right_change_count = 0

        self.reward_wrong_lane = env_config.get('reward_wrong_lane', -1.)

        # this data structure will contain all sensor data
        self.sensor_interface = SensorInterface()
        self.waypointer = None

        self.action_space = gym.spaces.Box(
            low=np.array([-1., -1.], dtype=np.float32),
            high=np.array([1., 1.], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(
                low=float('-inf'),
                high=float('inf'),
                # shape=(1, 3, 1, 3, 300, 300),
                shape=(1, 3, 3, 300, 300),
                dtype=np.float32,),
            gym.spaces.Box(
                low=float('-inf'),
                high=float('inf'),
                shape=(1, 1, 6),
                dtype=np.float32,),
            gym.spaces.Box(
                low=float('-inf'),
                high=float('inf'),
                shape=(1, 1, 1),
                dtype=np.float32,),
        ))

        # just to init the config
        self.setup_model(path_to_conf_file)

        # add sensors if doesn't exist
        if env_config.get('sensors') is None:
            env_config['sensors'] = self.sensors()

        # init env
        self.env = Environment(env_config)

    def reset(self, *, seed=None, options=None):
        self._global_plan = None
        self._global_plan_world_coord = None
        self.sensor_interface = None
        self.input_data = None
        self.waypointer = None

        state, info = self.env.reset(seed=seed, options=options)
        self.sensor_interface = self.env.sensor_interface
        self.set_world(CarlaDataProvider.get_world())
        self.set_global_plan(self.env.world_handler.gps_route, self.env.world_handler.vehicle_route)

        self.input_data = state
        state = self.run_step()

        return state, info

    def step(self, action):
        # action is the output of the model
        action = self.process_control_outputs(action)
        steer, throttle, brake = action
        steer = float(steer)
        throttle = float(throttle)
        brake = float(brake)

        state, reward, terminated, truncated, info = self.env.step([throttle, brake, steer])

        self.input_data = state
        state = self.run_step()

        reward += self.additional_reward()

        return state, reward, terminated, truncated, info

    def run_step(self):
        self.norm_rgb = [[self.process_image(self.input_data[camera_type][1]) for camera_type in g_conf.DATA_USED]]
        self.norm_speed = [np.float32([self.process_speed(self.input_data['SPEED'][1]['speed'])])]
        if g_conf.DATA_COMMAND_ONE_HOT:
            self.direction = \
                [np.float32(self.process_command(self.input_data['GPS'][1], self.input_data['IMU'][1])[0])]

        else:
            self.direction = \
                [np.float32([self.process_command(self.input_data['GPS'][1], self.input_data['IMU'][1])[1]-1])]

        self.norm_rgb = np.asarray(self.norm_rgb, dtype=np.float32)
        self.direction = np.asarray(self.direction, dtype=np.float32)
        self.norm_speed = np.asarray(self.norm_speed, dtype=np.float32)

        return self.norm_rgb, self.direction, self.norm_speed

    def close(self):
        self.norm_rgb = None
        self.norm_speed = None
        self.direction = None
        self.checkpoint = None
        self.world = None
        self.env.close()

    def additional_reward(self):
        # if the vehicle is out of road don't add any reward
        if self.env.out_of_road:
            return 0.

        command = self.waypointer._global_route[0][1]
        if command not in [RoadOption.LEFT, RoadOption.RIGHT]:
            if any([wp.is_junction for wp in self.env.bbox_wp]):
                return 0.

            # waypoint of the current plan
            plan_wp = self.env.map.get_waypoint(self.waypointer._global_route[0][0].location)
            lane_change_left = self.direction[0][4] > 0
            lane_change_right = self.direction[0][5] > 0
            if lane_change_left:
                self.left_change_count = 10
            elif self.left_change_count > 0:
                self.left_change_count -= 1
            if lane_change_right:
                self.right_change_count = 10
            elif self.right_change_count > 0:
                self.right_change_count -= 1

            # for the front corners
            for i, wp in enumerate(self.env.bbox_wp[:2]):
                # check out of lane
                if wp.lane_id != plan_wp.lane_id:
                    if wp.lane_type is not carla.LaneType.Driving: continue
                    if lane_change_left and abs(wp.lane_id) == abs(plan_wp.lane_id)+1:
                        continue
                    if lane_change_right and abs(wp.lane_id)+1 == abs(plan_wp.lane_id):
                        continue

                    if command in [RoadOption.STRAIGHT, RoadOption.LANEFOLLOW]:
                        # if we had a left change: for the right front corner check if it still on the other lane
                        if self.left_change_count > 0 and i == 1 and abs(wp.lane_id) == abs(plan_wp.lane_id)+1:
                            continue
                        # if we had a right change: for the left front corner check if it still on the other lane
                        if self.right_change_count > 0 and i == 0 and abs(wp.lane_id)+1 == abs(plan_wp.lane_id):
                            continue
                    return self.reward_wrong_lane

        return 0.

    def process_image(self, image):
        image = image[:,:,:3][:, :, ::-1] / 255.
        # to numpy array and normilize
        image = np.asarray(image, dtype=np.float32)
        image = (image - g_conf.IMG_NORMALIZATION['mean']) / g_conf.IMG_NORMALIZATION['std']
        image = image.transpose((2, 0 , 1))
        return image

    def process_speed(self, speed):
        norm_speed = abs(speed - g_conf.DATA_NORMALIZATION['speed'][0]) / (
                g_conf.DATA_NORMALIZATION['speed'][1] - g_conf.DATA_NORMALIZATION['speed'][0])  # [0.0, 1.0]
        return norm_speed

    def process_control_outputs(self, action_outputs):
        if g_conf.ACCELERATION_AS_ACTION:
            steer, self.acceleration = action_outputs[0], action_outputs[1]
            if self.acceleration >= 0.0:
                throttle = self.acceleration
                brake = 0.0
            else:
                brake = np.abs(self.acceleration)
                throttle = 0.0
        else:
            steer, throttle, brake = action_outputs[0], action_outputs[1], action_outputs[2]
            if brake < 0.05:
                brake = 0.0

        return np.clip(steer, -1, 1), np.clip(throttle, 0, 1), np.clip(brake, 0, 1)

    def process_command(self, gps, imu):
        if g_conf.DATA_COMMAND_CLASS_NUM == 6:
            _, _, cmd = self.waypointer.tick_lb(gps, imu)
            return encode_directions_6(cmd.value), cmd.value
        else:
            logger.fatal('g_conf.DATA_COMMAND_CLASS_NUM != 6 not supported')
            exit(1)

    def setup_model(self, path_to_conf_file):
        exp_dir = os.path.split(path_to_conf_file)[0]

        yaml_conf, _, _ = checkpoint_parse_configuration_file(path_to_conf_file)
        g_conf.immutable(False)
        merge_with_yaml(os.path.join(exp_dir, yaml_conf), process_type='drive')

        root_dir = os.path.join(*os.path.normpath(exp_dir).split(os.path.sep)[:-3])
        if exp_dir[0] == os.path.sep:
            root_dir = os.path.sep + root_dir
        set_type_of_process('train_only', root=root_dir)

    def set_world(self, world):
        self.world = world

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """

        ds_ids = downsample_route(global_plan_world_coord, 50)
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]
        self.waypointer = Waypointer(self.world, global_plan_gps=self._global_plan, global_route=global_plan_world_coord)

    def sensors(self):
        sensors = [
            {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 300, 'height': 300, 'fov': 60, 'id': 'rgb_central', 'lens_circle_setting': False},

            {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -60,
             'width': 300, 'height': 300, 'fov': 60, 'id': 'rgb_left', 'lens_circle_setting': False},

            {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
             'width': 300, 'height': 300, 'fov': 60, 'id': 'rgb_right', 'lens_circle_setting': False},

            {'type': 'sensor.other.gnss', 'id': 'GPS'},

            {'type': 'sensor.other.imu', 'id': 'IMU'},

            {'type': 'sensor.speedometer', 'id': 'SPEED'},
        ]
        return sensors

