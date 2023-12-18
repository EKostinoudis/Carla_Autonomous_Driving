#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the base class for all autonomous agents
"""
import carla
import os
import torch
import json
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image

from srunner.tools.route_manipulation import downsample_route
from srunner.tools.route_manipulation import interpolate_trajectory

from configs import g_conf, merge_with_yaml, set_type_of_process
from network.models_console import Models
from dataloaders.transforms import encode_directions_4, encode_directions_6

from waypointer import Waypointer

from importlib import import_module

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


def checkpoint_parse_configuration_file(filename):
    with open(filename, 'r') as f:
        configuration_dict = json.loads(f.read())

    return configuration_dict['yaml'], configuration_dict['checkpoint'], \
           configuration_dict['agent_name']

def load_entry_point(name):
    mod_name, attr_name = name.split(":")
    mod = import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn

def get_entry_point():
    return 'CILv2_agent'

class CILv2_agent(AutonomousAgent):
    """
    Autonomous agent base class. All user agents have to be derived from this class
    """

    def setup(self,
              path_to_conf_file2,
              path_to_conf_file='_results/Ours/Town12346_5/config40.json',
              save_driving_vision=False,
              save_driving_measurement=False,
              device=None):
        directoy = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(*(path_to_conf_file.split('/')))
        path_to_conf_file = os.path.join(directoy, path)
        self.track = Track.SENSORS
        #  current global plans to reach a destination
        self._global_plan = None
        self._global_plan_world_coord = None

        self.device = device
        if self.device is None: self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # this data structure will contain all sensor data
        # self.waypointer = None
        self.vision_save_path = save_driving_vision
        self.save_measurement = save_driving_measurement

        # agent's initialization
        self.setup_model(path_to_conf_file)


    def setup_model(self, path_to_conf_file):
        """
        Initialize everything needed by your agent and set the track attribute to the right type:
            Track.SENSORS : CAMERAS, LIDAR, RADAR, GPS and IMU sensors are allowed
            Track.MAP : OpenDRIVE map is also allowed
        """

        # exp_dir = os.path.join('/', os.path.join(*path_to_conf_file.split('/')[:-1]))
        # exp_dir = os.path.join(os.path.join(*path_to_conf_file.split('/')[:-1]))
        exp_dir = os.path.split(path_to_conf_file)[0]
        yaml_conf, checkpoint_number, _ = checkpoint_parse_configuration_file(path_to_conf_file)
        g_conf.immutable(False)
        merge_with_yaml(os.path.join(exp_dir, yaml_conf), process_type='drive')
        # set_type_of_process('drive', root=os.environ["TRAINING_RESULTS_ROOT"])
        root = os.path.split(exp_dir)[0]
        root = os.path.split(root)[0]
        root = os.path.split(root)[0]
        set_type_of_process('drive',
                            # root=os.path.join(os.path.join(*exp_dir.split('/')[:-3])))
                            root=root)

        self._model = Models(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)

        self.checkpoint = torch.load(
            os.path.join(exp_dir, 'checkpoints', self._model.name + '_' + str(checkpoint_number) + '.pth'),
            map_location=self.device,
            )
        self._model.load_state_dict(self.checkpoint['model'])
        self._model.eval()
        self._model.to(self.device)

    def set_world(self, world):
        self.world = world

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """

        ds_ids = downsample_route(global_plan_world_coord, 50)
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]
        self.waypointer = Waypointer(CarlaDataProvider.get_world(), global_plan_gps=self._global_plan, global_route=global_plan_world_coord)

    def set_waypointer(self):
        """
        Set the plan (route) for the agent
        """
        # global_plan_world_coord = self._global_plan
        # global_plan_gps = self._global_plan_world_coord
        # ds_ids = downsample_route(global_plan_world_coord, 50)
        # self.global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
        # self.global_plan = [global_plan_gps[x] for x in ds_ids]
        #
        # # cheat here
        self.waypointer = Waypointer(CarlaDataProvider.get_world(),
                                     global_plan_gps=self._global_plan,
                                     global_route=self._global_plan_world_coord)


    def sensors(self):  # pylint: disable=no-self-use
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        """
        sensors = [
            {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 300, 'height': 300, 'fov': 60, 'id': 'rgb_central', 'lens_circle_setting': False},

            {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -60,
             'width': 300, 'height': 300, 'fov': 60, 'id': 'rgb_left', 'lens_circle_setting': False},

            {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
             'width': 300, 'height': 300, 'fov': 60, 'id': 'rgb_right', 'lens_circle_setting': False},

            {'type': 'sensor.other.gnss', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'id': 'GPS'},

            {'type': 'sensor.other.imu', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'id': 'IMU'},

            {'type': 'sensor.speedometer', 'id': 'SPEED'},

            # {'type': 'sensor.opendrive_map', 'reading_frequency': 1, 'id': 'map'},

            # {'type': 'sensor.can_bus', 'id': 'can_bus'}
        ]

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        :return: control
        """
        if self.waypointer is None:
            self.set_waypointer()
        self.input_data = input_data

        self.control = carla.VehicleControl()
        self.norm_rgb = [[self.process_image(self.input_data[camera_type][1]).unsqueeze(0).to(self.device) for camera_type in g_conf.DATA_USED]]
        self.norm_speed = [torch.FloatTensor([self.process_speed(self.input_data['SPEED'][1]['speed'])]).unsqueeze(0)]
        if g_conf.DATA_COMMAND_ONE_HOT:
            self.direction = \
                [torch.FloatTensor(self.process_command(self.input_data['GPS'][1], self.input_data['IMU'][1])[0]).unsqueeze(0).to(self.device)]

        else:
            self.direction = \
                [torch.LongTensor([self.process_command(self.input_data['GPS'][1], self.input_data['IMU'][1])[1]-1]).unsqueeze(0).to(self.device)]

        # actions_outputs, _, self.attn_weights = self._model.forward_eval(self.norm_rgb, self.direction, self.norm_speed)
        actions_outputs = self._model.forward(self.norm_rgb, self.direction, self.norm_speed)

        action_outputs = self.process_control_outputs(actions_outputs.detach().cpu().numpy().squeeze())

        self.steer, self.throttle, self.brake = action_outputs
        self.control.steer = float(self.steer)
        self.control.throttle = float(self.throttle)
        self.control.brake = float(self.brake)
        self.control.hand_brake = False
        return self.control

    def destroy(self):
        """
        Destroy (clean-up) the agent
        :return:
        """
        self._model = None
        self.norm_rgb = None
        self.norm_speed = None
        self.direction = None
        self.checkpoint = None
        self.world = None
        # self.attn_weights = None

        self.reset()

    def reset(self):
        self.track = Track.SENSORS
        self._global_plan = None
        self._global_plan_world_coord = None
        self.input_data = None
        self.waypointer = None

    def process_image(self, image):
        # image = Image.fromarray(image)
        image = Image.fromarray(image[:,:,:3][:, :, ::-1])
        # image = Image.fromarray(image[:,:,:3])
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

    def process_command(self, gps, imu):
        if g_conf.DATA_COMMAND_CLASS_NUM == 4:
            _, _, cmd = self.waypointer.tick_nc(gps, imu)
            return encode_directions_4(cmd.value), cmd.value
        elif g_conf.DATA_COMMAND_CLASS_NUM == 6:
            _, _, cmd = self.waypointer.tick_lb(gps, imu)
            return encode_directions_6(cmd.value), cmd.value
 
