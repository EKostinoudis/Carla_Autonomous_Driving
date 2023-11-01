import os
import torch
import json
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from omegaconf import DictConfig
import gymnasium as gym

from environment.sensor_interface import SensorInterface
from srunner.tools.route_manipulation import downsample_route
from configs import g_conf, merge_with_yaml, set_type_of_process
from dataloaders.transforms import encode_directions_4, encode_directions_6

from .waypointer import Waypointer
from environment import Environment
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


def checkpoint_parse_configuration_file(filename):
    with open(filename, 'r') as f:
        configuration_dict = json.loads(f.read())

    return configuration_dict['yaml'], configuration_dict['checkpoint'], \
           configuration_dict['agent_name']

class CILv2_env(gym.Env):
    def __init__(self, env_config: DictConfig, path_to_conf_file):
        #  current global plans to reach a destination
        self._global_plan = None
        self._global_plan_world_coord = None

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

        return state, reward, terminated, truncated, info

    def run_step(self):
        # self.norm_rgb = [[self.process_image(self.input_data[camera_type][1]).unsqueeze(0) for camera_type in g_conf.DATA_USED]]
        # self.norm_rgb = [[self.process_image(self.input_data[camera_type][1])[np.newaxis, ...] for camera_type in g_conf.DATA_USED]]
        self.norm_rgb = [[self.process_image(self.input_data[camera_type][1]) for camera_type in g_conf.DATA_USED]]
        # self.norm_speed = [torch.FloatTensor([self.process_speed(self.input_data['SPEED'][1]['speed'])]).unsqueeze(0)]
        self.norm_speed = [np.float32([self.process_speed(self.input_data['SPEED'][1]['speed'])])[np.newaxis, ...]]
        if g_conf.DATA_COMMAND_ONE_HOT:
            '''
            self.direction = \
                [torch.FloatTensor(self.process_command(self.input_data['GPS'][1], self.input_data['IMU'][1])[0]).unsqueeze(0)]
            '''
            self.direction = \
                [np.float32(self.process_command(self.input_data['GPS'][1], self.input_data['IMU'][1])[0])[np.newaxis, ...]]

        else:
            '''
            self.direction = \
                [torch.LongTensor([self.process_command(self.input_data['GPS'][1], self.input_data['IMU'][1])[1]-1]).unsqueeze(0)]
            '''
            self.direction = \
                [np.float32([self.process_command(self.input_data['GPS'][1], self.input_data['IMU'][1])[1]-1])[np.newaxis, ...]]
        # actions_outputs = self._model.forward(self.norm_rgb, self.direction, self.norm_speed)

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

    def process_image(self, image):
        image2 = image[:,:,:3][:, :, ::-1] / 255.
        '''
        image = Image.fromarray(image[:,:,:3][:, :, ::-1])
        image = image.resize((g_conf.IMAGE_SHAPE[2], g_conf.IMAGE_SHAPE[1])).convert('RGB')
        image = TF.to_tensor(image)
        # Normalization is really necessary if you want to use any pretrained weights.
        image = TF.normalize(image, mean=g_conf.IMG_NORMALIZATION['mean'], std=g_conf.IMG_NORMALIZATION['std'], inplace=False)
        '''

        # to numpy array and normilize
        image2 = np.asarray(image2, dtype=np.float32)
        image2 = (image2 - g_conf.IMG_NORMALIZATION['mean']) / g_conf.IMG_NORMALIZATION['std']
        image2 = image2.transpose((2, 0 , 1))
        return image2

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
        if g_conf.DATA_COMMAND_CLASS_NUM == 4:
            _, _, cmd = self.waypointer.tick_nc(gps, imu)
            return encode_directions_4(cmd.value), cmd.value
        elif g_conf.DATA_COMMAND_CLASS_NUM == 6:
            _, _, cmd = self.waypointer.tick_lb(gps, imu)
            return encode_directions_6(cmd.value), cmd.value

    def setup_model(self, path_to_conf_file):
        exp_dir = os.path.split(path_to_conf_file)[0]

        yaml_conf, _, _ = checkpoint_parse_configuration_file(path_to_conf_file)
        g_conf.immutable(False)
        merge_with_yaml(os.path.join(exp_dir, yaml_conf), process_type='drive')
        set_type_of_process('train_only', root=os.path.join(*os.path.normpath(exp_dir).split(os.path.sep)[:-3]))


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

