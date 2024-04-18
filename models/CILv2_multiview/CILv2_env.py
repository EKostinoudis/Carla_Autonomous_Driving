from __future__ import annotations

import os
import json
import numpy as np
from omegaconf import DictConfig
import gymnasium as gym
import logging
from omegaconf import DictConfig, OmegaConf
from ray.rllib.env.env_context import EnvContext
from typing import Optional
import random
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from configs import g_conf, merge_with_yaml, set_type_of_process
from dataloaders.transforms import encode_directions_6

from environment import Environment

logger = logging.getLogger(__name__)

def checkpoint_parse_configuration_file(filename):
    with open(filename, 'r') as f:
        configuration_dict = json.loads(f.read())

    return configuration_dict['yaml'], configuration_dict['checkpoint'], \
           configuration_dict['agent_name']

CILv2_observation_space = gym.spaces.Tuple((
    gym.spaces.Box(
        low=float('-inf'),
        high=float('inf'),
        # shape=(1, 3, 1, 3, 300, 300),
        shape=(1, 3, 3, 300, 300),
        dtype=np.float32,),
    gym.spaces.Box(
        low=float('-inf'),
        high=float('inf'),
        shape=(1, 6),
        dtype=np.float32,),
    gym.spaces.Box(
        low=float('-inf'),
        high=float('inf'),
        shape=(1, 1),
        dtype=np.float32,),
))

class CILv2_env(gym.Env):
    def __init__(self,
                 env_config: DictConfig | dict,
                 path_to_conf_file: str,
                 rllib_config: Optional[EnvContext] = None,
                 ):
        # for the RLlib training:
        # update the port so every worker has different port
        if rllib_config is not None:
            offset = rllib_config.worker_index - 1
            seed = env_config.get('seed', random.randint(0, 10000)) + offset
            port = env_config.get('port', 2000) + 4*offset
            tm_port = env_config.get('tm_port', 8000) + offset
            env_config.update({'port': port, 'tm_port': tm_port, 'seed': seed})

        if not isinstance(env_config, DictConfig):
            env_config = OmegaConf.create(dict(env_config))

        self.return_reward_info = env_config.get('return_reward_info', False)

        self.left_change_count = 0
        self.right_change_count = 0

        self.action_space = gym.spaces.Box(
            low=np.array([-1., -1.], dtype=np.float32),
            high=np.array([1., 1.], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = CILv2_observation_space

        # just to init the config
        self.setup_model(path_to_conf_file)

        # add sensors if doesn't exist
        if env_config.get('sensors') is None:
            env_config['sensors'] = self.sensors()

        # init env
        self.env = None
        self.env_config = env_config
        self.restart_env()

    def reset(self, *, seed=None, options=None):
        self.input_data = None

        state, info = self.env.reset(seed=seed, options=options)

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

    def restart_env(self):
        # kill carla server if it exists
        if self.env is not None and self.env.carla_launcher is not None: self.env.carla_launcher.kill()

        self.env = Environment(self.env_config)

    def run_step(self):
        self.norm_rgb = torch.stack(
            [self.process_image(self.input_data[camera_type][1]) \
                for camera_type in g_conf.DATA_USED],
        ).unsqueeze(0)

        self.norm_speed = torch.tensor(
            [self.process_speed(self.input_data['SPEED'][1]['speed'])],
            dtype=torch.float32,
        ).unsqueeze(0)

        self.direction = torch.tensor(
            encode_directions_6(self.env.navigation_commad.value),
            dtype=torch.float32,
        ).unsqueeze(0)

        return self.norm_rgb, self.direction, self.norm_speed

    def close(self):
        self.norm_rgb = None
        self.norm_speed = None
        self.direction = None
        self.checkpoint = None
        self.env.close()

    def process_image(self, image):
        image = Image.fromarray(image[:,:,:3][:, :, ::-1])
        image = image.resize((g_conf.IMAGE_SHAPE[2], g_conf.IMAGE_SHAPE[1])).convert('RGB')
        image = TF.to_tensor(image)

        # Normalization is really necessary if you want to use any pretrained weights.
        image = TF.normalize(image, mean=g_conf.IMG_NORMALIZATION['mean'], std=g_conf.IMG_NORMALIZATION['std'])
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

    def setup_model(self, path_to_conf_file):
        exp_dir = os.path.split(path_to_conf_file)[0]

        yaml_conf, _, _ = checkpoint_parse_configuration_file(path_to_conf_file)
        g_conf.immutable(False)
        merge_with_yaml(os.path.join(exp_dir, yaml_conf), process_type='drive')

        root_dir = os.path.abspath(os.path.join(exp_dir, '..', '..', '..'))
        set_type_of_process('train_only', root=root_dir)

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

