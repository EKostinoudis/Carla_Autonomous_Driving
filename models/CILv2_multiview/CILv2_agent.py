#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the base class for all autonomous agents
"""

from __future__ import print_function

from enum import Enum

import carla
import os
import torch
import json
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
# import plotly.graph_objects as go

from environment.sensor_interface import SensorInterface
from srunner.tools.route_manipulation import downsample_route
from srunner.tools.route_manipulation import interpolate_trajectory

from configs import g_conf, merge_with_yaml, set_type_of_process
from network.models_console import Models
from _utils.training_utils import DataParallelWrapper
from dataloaders.transforms import encode_directions_4, encode_directions_6, inverse_normalize, decode_directions_4, \
    decode_directions_6

from .waypointer import Waypointer

from importlib import import_module


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


class Track(Enum):
    """
    This enum represents the different tracks of the CARLA AD leaderboard.
    """
    SENSORS = 'SENSORS'
    MAP = 'MAP'


class CILv2_agent(object):
    """
    Autonomous agent base class. All user agents have to be derived from this class
    """

    def __init__(self, path_to_conf_file, save_driving_vision, save_driving_measurement, device=None):
        self.track = Track.SENSORS
        #  current global plans to reach a destination
        self._global_plan = None
        self._global_plan_world_coord = None

        self.device = device
        if self.device is None: self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # this data structure will contain all sensor data
        self.sensor_interface = SensorInterface()
        self.waypointer = None
        # self.attn_weights = None
        self.vision_save_path = save_driving_vision
        self.save_measurement = save_driving_measurement

        # agent's initialization
        self.setup_model(path_to_conf_file)

        self.cmap_2 = plt.get_cmap('jet')
        self.cmap_1 = plt.get_cmap('Reds')
        self.datapoint_count = 0
        self.save_frequence = 1

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
        set_type_of_process('drive',
                            # root=os.path.join(os.path.join(*exp_dir.split('/')[:-3])))
                            root=os.path.join(*os.path.normpath(exp_dir).split(os.path.sep)[:-3]))

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
        self.waypointer = Waypointer(self.world, global_plan_gps=self._global_plan, global_route=global_plan_world_coord)

    '''
    def reset_global_plan(self):
        """
        reset the plan (route) for the agent
        """
        current_loc = self._ego_vehicle.get_location()
        last_gps, _ = self._global_plan[-1]
        last_loc = self.waypointer.gps_to_location([last_gps['lat'], last_gps['lon'], last_gps['z']])
        gps_route, route = interpolate_trajectory(self.world, [current_loc, last_loc])

        ds_ids = downsample_route(route, 50)
        self.route = [(route[x][0], route[x][1]) for x in ds_ids]
        self._global_plan = [gps_route[x] for x in ds_ids]
        self.waypointer.reset_route(global_plan_gps=self._global_plan, global_route=route)
        return route
    '''

    '''
    def set_ego_vehicle(self, ego_vehicle):
        self._ego_vehicle = ego_vehicle
    '''

    def sensors(self):  # pylint: disable=no-self-use
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        """

        if self.vision_save_path:
            sensors = [
                {'type': 'sensor.camera.rgb', 'x': -4.5, 'y': 0.0, 'z': 4.0, 'roll': 0.0, 'pitch': -20.0, 'yaw': 0.0,
                 'width': 1088, 'height': 680, 'fov': 120, 'id': 'rgb_backontop', 'lens_circle_setting': False},

                {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                 'width': 300, 'height': 300, 'fov': 60, 'id': 'rgb_central', 'lens_circle_setting': False},

                {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -60,
                 'width': 300, 'height': 300, 'fov': 60, 'id': 'rgb_left', 'lens_circle_setting': False},

                {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                 'width': 300, 'height': 300, 'fov': 60, 'id': 'rgb_right', 'lens_circle_setting': False},

                {'type': 'sensor.other.gnss', 'id': 'GPS'},

                {'type': 'sensor.other.imu', 'id': 'IMU'},

                {'type': 'sensor.speedometer', 'id': 'SPEED'},

                # {'type': 'sensor.can_bus', 'id': 'can_bus'}
            ]

        else:
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

                # {'type': 'sensor.can_bus', 'id': 'can_bus'}
            ]

        return sensors

    def __call__(self, timestamp):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """
        self.input_data = self.sensor_interface.get_data()

        # for key, values in self.input_data.items():
        #     if values[0] != timestamp.frame:
        #         raise RuntimeError(' The frame number of sensor data does not match the timestamp frame:', key)

        control = self.run_step()
        control.manual_gear_shift = False

        return control

    def step(self, input_data):
        self.input_data = input_data
        control = self.run_step()
        control.manual_gear_shift = False

        return control

    def run_step(self):
        """
        Execute one step of navigation.
        :return: control
        """

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

        '''
        if self.vision_save_path:
            self.record_driving(self.input_data)
        '''

        self.datapoint_count += 1

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
        self.sensor_interface = None
        self.input_data = None
        self.waypointer = None
        self.vision_save_path = None
        self.datapoint_count = 0

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

    '''
    def record_driving(self, current_input_data):
        if not os.path.exists(self.vision_save_path):
            os.makedirs(self.vision_save_path)
        if self.datapoint_count % self.save_frequence == 0:
            cams = []
            for i in range(len(g_conf.DATA_USED)):
                rgb_img = inverse_normalize(self.norm_rgb[-1][i], g_conf.IMG_NORMALIZATION['mean'], g_conf.IMG_NORMALIZATION['std']).detach().cpu().numpy().squeeze(0)
                cams.append(Image.fromarray((rgb_img.transpose(1, 2, 0) * 255).astype(np.uint8)))

            rgb_backontop = Image.fromarray(current_input_data['rgb_backontop'][1])

            if g_conf.DATA_COMMAND_ONE_HOT:
                cmd = decode_directions_6(self.direction[-1].detach().cpu().numpy().squeeze(0))
            else:
                cmd = self.direction[-1].detach().cpu().numpy().squeeze(0) + 1
            if float(cmd) == 1.0:
                command_sign = Image.open(os.path.join(os.getcwd(), 'signs', '6_directions', 'turn_left.png'))

            elif float(cmd) == 2.0:
                command_sign = Image.open(os.path.join(os.getcwd(), 'signs', '6_directions', 'turn_right.png'))

            elif float(cmd) == 3.0:
                command_sign = Image.open(os.path.join(os.getcwd(), 'signs', '6_directions', 'go_straight.png'))

            elif float(cmd) == 4.0:
                command_sign = Image.open(os.path.join(os.getcwd(), 'signs', '6_directions', 'follow_lane.png'))

            elif float(cmd) == 5.0:
                command_sign = Image.open(os.path.join(os.getcwd(), 'signs', '6_directions', 'change_left.png'))

            elif float(cmd) == 6.0:
                command_sign = Image.open(os.path.join(os.getcwd(), 'signs', '6_directions', 'change_right.png'))

            else:
                raise RuntimeError()

            command_sign = command_sign.resize((280, 71))

            mat = Image.new('RGB', (rgb_backontop.width+len(cams)*(cams[0].width + 10), 120+ rgb_backontop.height), (0, 0, 0))
            draw_mat = ImageDraw.Draw(mat)
            font = ImageFont.truetype(os.path.join(os.getcwd(), 'signs', 'arial.ttf'), 30)
            font_2 = ImageFont.truetype(os.path.join(os.getcwd(), 'signs', 'arial.ttf'), 55)
            font_3 = ImageFont.truetype(os.path.join(os.getcwd(), 'signs', 'arial.ttf'), 25)

            # third person
            draw_mat.text((260 , 40), str("Third Person Perspective"), fill=(255, 255, 255), font=font_2)
            mat.paste(rgb_backontop, (0, 120))

            # RGB
            draw_mat.text((330 + rgb_backontop.width, 30), str("RGB Cameras Input"), fill=(255, 255, 255), font=font)
            draw_mat.text((120 + rgb_backontop.width, 80), str("-60.0"+'\u00b0'), fill=(255, 255, 255), font=font)
            draw_mat.text((435 + rgb_backontop.width, 80), str("0.0"+'\u00b0'), fill=(255, 255, 255), font=font)
            draw_mat.text((735 + rgb_backontop.width, 80), str("60.0"+'\u00b0'), fill=(255, 255, 255), font=font)
            mat.paste(cams[0], (rgb_backontop.width+10, 120))
            mat.paste(cams[1], (rgb_backontop.width+10 +int(cams[0].width) + 10, 120))
            mat.paste(cams[2], (rgb_backontop.width+10 + int((cams[0].width) + 10)*2, 120))

            # command
            draw_mat.text((rgb_backontop.width+125, int(rgb_backontop.height/2)+130), str("Command Input"), fill=(255, 255, 255), font=font)
            mat.paste(command_sign, (rgb_backontop.width+100, int(rgb_backontop.height/2) + 170))

            # speed
            draw_mat.text((rgb_backontop.width+580, int(rgb_backontop.height/2)+130), str("Speed Input (m/s)"), fill=(255, 255, 255), font=font)
            speed_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                gauge={'axis': {'range': [0, 12], 'visible': False},
                       'bordercolor': "white",
                       'borderwidth': 3},
                value= round(current_input_data['SPEED'][1]['speed'], 3),
                domain={'x': [0, 1], 'y': [0, 1]},))
            speed_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "white", 'family': "Arial", "size": 3})
            buf = io.BytesIO()
            speed_gauge.write_image(buf, format='png')
            buf.seek(0)
            speed_gauge = Image.open(buf)
            speed_gauge = speed_gauge.resize((162, 117))
            speed_gauge = speed_gauge.crop((10, 20, 182, 124))
            mat.paste(speed_gauge, (rgb_backontop.width + 630, int(rgb_backontop.height/2)+170))
            draw_mat.text((rgb_backontop.width+610, int(rgb_backontop.height/2) + 215), str("0" ), fill=(255, 255, 255), font=font_3)
            draw_mat.text((rgb_backontop.width+775, int(rgb_backontop.height/2) + 215), str("12"), fill=(255, 255, 255), font=font_3)

            # steer
            draw_mat.text((rgb_backontop.width + 50, 120 + int(rgb_backontop.height/2) + 180),
                          str("Steering Prediction  "+ "%.3f" % np.clip(self.steer, -1, 1)),
                          fill=(255, 255, 255),
                          font=font)
            if self.steer > 0.0:
                step = [
                    {'range': [-1, 0], 'color': "black"},
                    {'range': [0, self.steer], 'color': "yellow"},
                    {'range': [self.steer, 1], 'color': "black"}]
            else:
                step = [
                    {'range': [-1, self.steer], 'color': "black"},
                    {'range': [self.steer, 0], 'color': "yellow"},
                    {'range': [0, 1], 'color': "black"}]
            steer_gauge = go.Figure(go.Indicator(
                mode="number+gauge",
                gauge={'shape': "bullet",
                       'axis': {'range': [-1, 1], 'visible': False},
                       'bordercolor': "white",
                       'borderwidth': 3,
                       'steps': step,
                       'threshold': {
                           'line': {'color': "white", 'width': 3},
                           'thickness': 1.0, 'value': 0},},
                domain={'x': [0, 1], 'y': [0, 1]}))
            steer_gauge.update_layout(
                height = 250,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "white", 'family': "Arial", "size": 3})
            buf = io.BytesIO()
            steer_gauge.write_image(buf, format='png')
            buf.seek(0)
            steer_gauge = Image.open(buf)
            steer_gauge = steer_gauge.resize((359, 128))
            steer_gauge = steer_gauge.crop((40, 40, 250, 100))
            mat.paste(steer_gauge, (rgb_backontop.width+120 , 120 + int(rgb_backontop.height/2) + 230))
            draw_mat.text((rgb_backontop.width+70, 120 + int(rgb_backontop.height/2) + 240), str("-1" ), fill=(255, 255, 255), font=font)
            draw_mat.text((rgb_backontop.width+360, 120 + int(rgb_backontop.height/2) + 240), str("1"), fill=(255, 255, 255), font=font)

            # acceleration
            draw_mat.text((rgb_backontop.width+500 , 120 + int(rgb_backontop.height/2) + 180),
                          str("Acceleration Prediction  "+ "%.3f" % np.clip(self.acceleration, -1, 1)),
                          fill=(255, 255, 255),
                          font=font)
            if self.acceleration > 0.0:
                step = [
                    {'range': [-1, 0], 'color': "black"},
                    {'range': [0, self.acceleration], 'color': "orange"},
                    {'range': [self.acceleration, 1], 'color': "black"}]
            else:
                step = [
                    {'range': [-1, self.acceleration], 'color': "black"},
                    {'range': [self.acceleration, 0], 'color': "lightgray"},
                    {'range': [0, 1], 'color': "black"}]
            acc_gauge = go.Figure(go.Indicator(
                mode="number+gauge",
                gauge={'shape': "bullet",
                       'axis': {'range': [-1, 1], 'visible': False},
                       'bordercolor': "white",
                       'borderwidth': 3,
                       'steps': step,
                       'threshold': {
                           'line': {'color': "white", 'width': 3},
                           'thickness': 1.0, 'value': 0},},
                domain={'x': [0, 1], 'y': [0, 1]}))
            acc_gauge.update_layout(
                height = 250,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "white", 'family': "Arial", "size": 3})
            buf = io.BytesIO()
            acc_gauge.write_image(buf, format='png')
            buf.seek(0)
            acc_gauge = Image.open(buf)
            acc_gauge = acc_gauge.resize((359, 128))
            acc_gauge = acc_gauge.crop((40, 40, 250, 100))
            mat.paste(acc_gauge, (rgb_backontop.width+580 , 120 + int(rgb_backontop.height/2) + 230))
            draw_mat.text((rgb_backontop.width+530, 120 + int(rgb_backontop.height/2) + 240), str("-1" ), fill=(255, 255, 255), font=font)
            draw_mat.text((rgb_backontop.width+820, 120 + int(rgb_backontop.height/2) + 240), str("1"), fill=(255, 255, 255), font=font)

            mat = mat.resize((int(mat.size[0] / 3), int(mat.size[1] / 3)))   # comment this if you don't want to resize
            mat.save(os.path.join(self.vision_save_path, str(self.datapoint_count).zfill(6) + '.jpg'))
            if self.save_measurement:
                # we record the driving measurement data
                data = current_input_data['can_bus'][1]
                data.update({'steer': np.nan_to_num(self.control.steer)})
                data.update({'throttle': np.nan_to_num(self.control.throttle)})
                data.update({'brake': np.nan_to_num(self.control.brake)})
                data.update({'hand_brake': self.control.hand_brake})
                data.update({'reverse': self.control.reverse})
                data.update({'speed': current_input_data['SPEED'][1]['speed']})
                data.update({'direction': float(cmd)})
                with open(os.path.join(self.vision_save_path,
                                       'can_bus' + str(self.datapoint_count).zfill(6) + '.json'), 'w') as fo:
                    jsonObj = {}
                    jsonObj.update(data)
                    fo.seek(0)
                    fo.write(json.dumps(jsonObj, sort_keys=True, indent=4))
                    fo.close()
    '''
