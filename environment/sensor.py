import logging
import numpy as np
from collections import deque
import carla

from .sensor_interface import SpeedometerReader, CallBack

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

logger = logging.getLogger(__name__)

class Sensor():
    def __init__(self,
                 blueprint,
                 transform,
                 queue_size=1):
        self.blueprint = blueprint
        self.transform = transform
        self.queue_size = queue_size
        self.sensor = None

        # init data queue
        self.data = deque(maxlen=self.queue_size)

    def reset(self, vehicle, world, extract_data_fun=None):
        logger.debug(f'Reseting sensor: {self.blueprint.id}')

        # NOTE: throws an exception on failure
        self.sensor = world.spawn_actor(self.blueprint,
                                        self.transform,
                                        attach_to=vehicle)

        # set the callback function that extracts the data
        if extract_data_fun is not None:
            self.sensor.listen(lambda data: self.data.append(extract_data_fun(data)))
        else:
            self.sensor.listen(lambda data: self.data.append(data))

        self.data.clear() # make sure there is no data in queue

    def destroy(self):
        logger.debug(f'Destroying sensor: {self.blueprint.id}')
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None

        self.data.clear() # make sure there is no data in queue

def create_camera_sensor(bp_library, config):
    camera_bp = bp_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(config['width']))
    camera_bp.set_attribute('image_size_y', str(config['height']))
    camera_bp.set_attribute('fov', str(config['fov']))
    location = carla.Location(
            x=config['x'],
            y=config['y'],
            z=config['z'],)
    rotation = carla.Rotation(
            pitch=config['pitch'],
            roll=config['roll'],
            yaw=config['yaw'],)
    camera_transform = carla.Transform(location, rotation)
    return Sensor(camera_bp, camera_transform)

# Based on srunner/autoagents/agent_wrapper.py
def setup_sensors(sensors_list_conf, vehicle, sensor_interface, leaderboard_settings=False):
    bp_library = CarlaDataProvider.get_world().get_blueprint_library()

    sensors_list = []
    for sensor_spec in sensors_list_conf:
        if sensor_spec['type'].startswith('sensor.speedometer'):
            delta_time = CarlaDataProvider.get_world().get_settings().fixed_delta_seconds
            frame_rate = 1 / delta_time
            sensor = SpeedometerReader(vehicle, frame_rate)
        else:
            # These are the sensors spawned on the carla world
            bp = bp_library.find(str(sensor_spec['type']))
            if sensor_spec['type'].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(sensor_spec['width']))
                bp.set_attribute('image_size_y', str(sensor_spec['height']))
                bp.set_attribute('fov', str(sensor_spec['fov']))
                if leaderboard_settings:
                    bp.set_attribute('lens_circle_multiplier', str(3.0))
                    bp.set_attribute('lens_circle_falloff', str(3.0))
                    bp.set_attribute('chromatic_aberration_intensity', str(0.5))
                    bp.set_attribute('chromatic_aberration_offset', str(0))
                sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                 z=sensor_spec['z'])
                sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                 roll=sensor_spec['roll'],
                                                 yaw=sensor_spec['yaw'])
            elif sensor_spec['type'].startswith('sensor.lidar'):
                bp.set_attribute('range', str(sensor_spec['range']))
                bp.set_attribute('rotation_frequency', str(sensor_spec['rotation_frequency']))
                bp.set_attribute('channels', str(sensor_spec['channels']))
                bp.set_attribute('upper_fov', str(sensor_spec['upper_fov']))
                bp.set_attribute('lower_fov', str(sensor_spec['lower_fov']))
                bp.set_attribute('points_per_second', str(sensor_spec['points_per_second']))
                if leaderboard_settings:
                    bp.set_attribute('range', str(85))
                    bp.set_attribute('rotation_frequency', str(10))
                    bp.set_attribute('channels', str(64))
                    bp.set_attribute('upper_fov', str(10))
                    bp.set_attribute('lower_fov', str(-30))
                    bp.set_attribute('points_per_second', str(600000))
                    bp.set_attribute('atmosphere_attenuation_rate', str(0.004))
                    bp.set_attribute('dropoff_general_rate', str(0.45))
                    bp.set_attribute('dropoff_intensity_limit', str(0.8))
                    bp.set_attribute('dropoff_zero_intensity', str(0.4))
                sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                 z=sensor_spec['z'])
                sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                 roll=sensor_spec['roll'],
                                                 yaw=sensor_spec['yaw'])
            elif sensor_spec['type'].startswith('sensor.other.radar'):
                if leaderboard_settings:
                    bp.set_attribute('horizontal_fov', str(sensor_spec['fov']))  # degrees
                    bp.set_attribute('vertical_fov', str(sensor_spec['fov']))  # degrees
                    bp.set_attribute('points_per_second', '1500')
                    bp.set_attribute('range', '100')  # meters

                sensor_location = carla.Location(x=sensor_spec['x'],
                                                 y=sensor_spec['y'],
                                                 z=sensor_spec['z'])
                sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                 roll=sensor_spec['roll'],
                                                 yaw=sensor_spec['yaw'])
            elif sensor_spec['type'].startswith('sensor.other.gnss'):
                if leaderboard_settings:
                    bp.set_attribute('noise_alt_stddev', str(0.000005))
                    bp.set_attribute('noise_lat_stddev', str(0.000005))
                    bp.set_attribute('noise_lon_stddev', str(0.000005))
                    bp.set_attribute('noise_alt_bias', str(0.0))
                    bp.set_attribute('noise_lat_bias', str(0.0))
                    bp.set_attribute('noise_lon_bias', str(0.0))
                sensor_location = carla.Location()
                sensor_rotation = carla.Rotation()
            elif sensor_spec['type'].startswith('sensor.other.imu'):
                if leaderboard_settings:
                    bp.set_attribute('noise_accel_stddev_x', str(0.001))
                    bp.set_attribute('noise_accel_stddev_y', str(0.001))
                    bp.set_attribute('noise_accel_stddev_z', str(0.015))
                    bp.set_attribute('noise_gyro_stddev_x', str(0.001))
                    bp.set_attribute('noise_gyro_stddev_y', str(0.001))
                    bp.set_attribute('noise_gyro_stddev_z', str(0.001))
                sensor_location = carla.Location()
                sensor_rotation = carla.Rotation()

            # create sensor
            sensor_transform = carla.Transform(sensor_location, sensor_rotation)
            sensor = CarlaDataProvider.get_world().spawn_actor(bp, sensor_transform, vehicle)

        # setup callback
        sensor.listen(CallBack(sensor_spec['id'], sensor_spec['type'], sensor, sensor_interface))
        sensors_list.append(sensor)
    return sensors_list

def extract_rgb_data(data):
    height, width = data.height, data.width
    return np.frombuffer(data.raw_data, dtype=np.dtype("uint8")) \
            .reshape((height, width, 4))[:,:,:3]

def extract_rgb_data_factory(height, width):
    def extract_rgb_data(data):
        return np.frombuffer(data.raw_data, dtype=np.dtype("uint8")) \
                .reshape((height, width, 4))[:,:,:3]
    return extract_rgb_data

