import gymnasium as gym
import random
import numpy as np
import carla
import cv2
import logging
import math
from typing import Tuple
from omegaconf import DictConfig, OmegaConf
import torch

from .world_handler import WorldHandler
from .sensor import *
from .sensor_interface import SensorInterface
from .route_planner import RoutePlanner
from .carla_launcher import CarlaLauncher
from .dynamic_speed import DynamicSpeed

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    RunningStopTest,
    RunningRedLightTest,
    # WrongLaneTest,
    # OutsideRouteLanesTest,
    # OnSidewalkTest,
    # OffRoadTest,
)

from agents.navigation.local_planner import RoadOption

logger = logging.getLogger(__name__)


class Environment(gym.Env):
    def __init__(self, config: DictConfig):
        if not isinstance(config, DictConfig): config = OmegaConf.create(dict(config))
        # self.config = config
        self.sensors_config = config.sensors

        self.restart_server = False
        self.use_launcher = config.get('use_carla_launcher', False)
        if self.use_launcher:
            launch_script = config.get('carla_launch_script', None)
            if launch_script is None:
                raise ValueError('Must provide "carla_launch_script" in the environment config')
            self.carla_launcher = CarlaLauncher(
                config.get('port', 2000),
                launch_script,
                config.get('carla_restart_after', -1),
            )
        else:
            self.carla_launcher = None

        # how many times to repeat the given action a step of the env
        self.repeat_action = config.get('repeat_action', 0)

        # create world handler
        self.world_handler = WorldHandler(config)

        # set the seed
        seed = config.get('seed', 0)
        self.set_seed(seed)

        self.return_reward_info = config.get('return_reward_info', False)
        self.debug = config.get('debug', False)
        self.use_leaderboard_setting = config.get('use_leaderboard_setting', True)

        # display camera movement (mainly for debug)
        self.render_rgb_camera_flag = config.get('render_rgb_camera', False)
        
        # termination after stop sign or red light run
        self.termination_on_run = config.get('termination_on_run', True)
        self.fixed_delta_seconds = config.get('fixed_delta_seconds', 0.1)

        # maximum seconds that the vehicle can be stopped
        self.stopped_termination_seconds = config.get('stopped_termination_seconds', 90)

        # maximum seconds that the vehicle can be out of the lane, the junctions
        # don't count because itsn't posible
        self.out_of_lane_termination_seconds = config.get('out_of_lane_termination_seconds', 5)

        # get the reward constants-multipliers
        self.reward_failure = config.get('reward_failure', -10.)
        self.reward_success = config.get('reward_success', 10.)
        self.reward_wrong_lane = config.get('reward_wrong_lane', -1.)
        self.reward_steer = config.get('reward_steer', -0.1)
        self.reward_speed = config.get('reward_speed', 0.1)
        self.reward_max_speed = config.get('reward_max_speed', 30.)
        self.reward_speed_slope = config.get('reward_speed_slope', 1.)
        self.reward_waypoint = config.get('reward_waypoint', 30.)
        self.reward_speed_penalty = config.get('reward_speed_penalty', False)
        self.reward_dynamic_max_speed = config.get('reward_dynamic_max_speed', False)
        self.reward_negative_speed_overshoot = config.get('reward_negative_speed_overshoot', True)

        self.vehicle = None
        self.sensors_env = []
        self.sensors = []
        self.vehicle_control = carla.VehicleControl()

        self.dynamic_speed = None

        # initialize the sensors that are used in the environment
        self.init_env_sensors()

        # throttle, break, esteer
        self.action_space = gym.spaces.Box(
                low=np.array([0., 0., -1.], dtype=np.float32),
                high=np.array([1., 1., 1.], dtype=np.float32),
                dtype=np.float32,
                seed=seed,
                )

        # define observation space from sensors_config
        self.observation_space = self.create_observation_space()
        self.observation_space.seed(seed)

        # TODO: maybe specify the reward_range
        # self.reward_range = (min reward, max reward)

    def reset(self, seed=None, options=None):
        self.prev_steer = 0. # previoius steer value
        self.stopped_count = 0 # ticks the vehicle is stopped
        self.left_change_count = 0
        self.right_change_count = 0
        self.out_of_lane_count = 0
        self.in_junction_count = 0

        if seed is not None: self.set_seed(seed)

        self.destroy_sensors()

        # if restart is needed for the carla server (mainly to avoid memory leaks)
        if self.carla_launcher is not None:
            self.carla_launcher.reset(restart_server=self.restart_server)
            self.restart_server = False

        # reset the world and get the new vehicle
        self.world_handler.reset()
        self.vehicle = self.world_handler.vehicle

        # set the route
        self.route_planner = RoutePlanner(
            self.world_handler.gps_route,
            self.world_handler.vehicle_route,
        )

        # set the new map
        self.map = CarlaDataProvider.get_map()

        self.reset_env_sensors()

        # vehicle's sensors
        self.sensor_interface = SensorInterface()
        self.sensors = setup_sensors(
            self.sensors_config,
            self.vehicle,
            self.sensor_interface,
            self.use_leaderboard_setting,
        )

        self.init_tests()

        if self.reward_dynamic_max_speed:
            self.dynamic_speed = DynamicSpeed(
                self.vehicle,
                CarlaDataProvider.get_world(),
                self.map,
                self.reward_max_speed,
            )

        # reset controls
        self.vehicle_control.throttle = 0.
        self.vehicle_control.steer = 0.
        self.vehicle_control.brake = 0.

        CarlaDataProvider.get_world().tick()
        CarlaDataProvider.get_world().tick()

        new_state, reward, terminated, truncated, info = self.step([0, 0, 0])
        # NOTE: we return an empy dict because we don't log at reset
        return new_state, {}

    def step(self, action):
        ''' Performs one step of the simulation.
        1. Apply the given action.
        2. Calculate the reward.
        3. Check if the simulation ended.
        4. Get the new state.
        '''

        self.apply_action(action)

        for _ in range(self.repeat_action + 1):
            # world (server) tick
            CarlaDataProvider.get_world().tick()

            # update scenario
            self.episode_alive, self.task_failed = self.world_handler.step()

            # get teh new state
            new_state = self.get_state()

            # get the new command for navigation
            prev_len = len(self.route_planner.gps_route)
            self.navigation_commad = self.route_planner.step(
                new_state['GPS'][1],
                new_state['IMU'][1],
            )
            self.reached_wp = prev_len > len(self.route_planner.gps_route)

            if self.render_rgb_camera_flag: self.render_rgb_camera()

            # update the state of the tests
            for test in self.tests: _ = test.update()

            # check if the vehicle is out of road or lane
            self.in_junction = False
            self.out_of_road = self.check_out_of_road()
            self.out_of_lane = self.check_out_of_lane()
            if not self.in_junction:
                self.in_junction_count = 0
                if self.out_of_road or self.out_of_lane:
                    self.out_of_lane_count += 1
                else:
                    self.out_of_lane_count = 0
            else:
                if self.out_of_road or self.out_of_lane:
                    self.out_of_lane_count += 1
                self.in_junction_count += 1

            # update the stopped counter
            if self.get_velocity() < 0.5:
                self.stopped_count += 1
            else:
                self.stopped_count = 0

            if self.debug:
                if not self.episode_alive:
                    logger.debug(f'End episode. Task failed: {self.task_failed}')
                logger.debug(f'collision_detector: {self.collision_detector.data}')
                logger.debug(f'Out of road: {self.out_of_road}')
                logger.debug(f'Out of lane: {self.out_of_lane}')
                logger.debug(f'stopped count: {self.stopped_count}')
                logger.debug(f'Velocity: {self.get_velocity():6.02f} '
                             f'Speed limit: {self.vehicle.get_speed_limit():6.02f}')

            # calculate the reward
            reward = self.get_reward()

            # get the terminated and truncated values
            terminated, truncated = self.episode_end()

            if self.return_reward_info:
                info = self.create_info_for_logging()
            else:
                info = {}

            # if the episode terminated or truncated do not perform more steps
            if terminated or truncated: break

        # hold the previous steer value here, cause we only use it here
        self.prev_steer = self.vehicle_control.steer

        # returns (next state, reward, terminated, truncated, info)
        return new_state, reward, terminated, truncated, info

    def render(self): pass # currently empty

    def close(self):
        self.destroy_sensors()

        self.world_handler.close()

        if self.carla_launcher is not None: self.carla_launcher.kill()

    def apply_action(self, action):
        self.vehicle_control.throttle = action[0]
        self.vehicle_control.brake = action[1]
        self.vehicle_control.steer = action[2]

        # apply the action
        self.vehicle.apply_control(self.vehicle_control)

    def get_reward(self) -> float:
        self.episode_end_success_reward = 0.
        self.episode_end_fail_reward = 0.
        self.sign_run_reward = 0.
        self.not_moving_reward = 0.
        self.out_of_road_reward = 0.
        self.steering_reward = 0.
        self.speeding_reward = 0.
        self.collision_reward = 0.
        self.reached_wp_reward = self.reached_wp * self.reward_waypoint

        # end of scenario reward
        if not self.episode_alive:
            if self.task_failed:
                self.episode_end_fail_reward = self.reward_failure
                return self.episode_end_fail_reward
            else:
                self.episode_end_success_reward = self.reward_success
                return self.episode_end_success_reward

        if len(self.collision_detector.data) > 0:
            if self.reward_speed_penalty:
                self.collision_reward = self.reward_failure * (1 + self.get_velocity())
            else:
                self.collision_reward = self.reward_failure
            return self.collision_reward

        # stop and red light tests
        for test in self.tests[:2]:
            # print(test.name, test.test_status)
            if test.test_status == 'FAILURE':
                test.test_status = 'RESET'
                if self.reward_speed_penalty:
                    self.sign_run_reward = self.reward_failure * (1 + self.get_velocity())
                else:
                    self.sign_run_reward = self.reward_failure
                return self.sign_run_reward

        # vehicle too long stopped
        if self.stopped_count * self.fixed_delta_seconds > self.stopped_termination_seconds:
            self.not_moving_reward = self.reward_failure
            return self.not_moving_reward

        # vehicle too long out of lane
        if self.out_of_lane_count * self.fixed_delta_seconds > self.out_of_lane_termination_seconds:
            self.out_of_road_reward = self.reward_wrong_lane
            return self.reward_wrong_lane

        # out of road or lane
        if self.out_of_road or self.out_of_lane:
            self.out_of_road_reward = self.reward_wrong_lane

        # steering reward (based on steer diff)
        self.steering_reward = self.reward_steer * abs(self.prev_steer - self.vehicle_control.steer)

        # if the vehicle has speed lower than the given max, scale linearly the reward
        # else (above the speed limit), give penalty
        speed = self.get_velocity()
        if self.reward_dynamic_max_speed:
            desired_speed = self.dynamic_speed.get()
        else:
            desired_speed = self.reward_max_speed

        if self.reward_negative_speed_overshoot:
            if speed <= self.reward_max_speed:
                self.speeding_reward = self.reward_speed * (speed / desired_speed)
            else:
                self.speeding_reward = -self.reward_speed
        else:
            self.speeding_reward = self.reward_speed * \
                (1 - self.reward_speed_slope * abs(speed - desired_speed) / self.reward_max_speed)

        return self.not_moving_reward + \
               self.out_of_road_reward + \
               self.steering_reward + \
               self.speeding_reward + \
               self.reached_wp_reward

    def create_info_for_logging(self):
        return {
            'episode_end_success_reward': self.episode_end_success_reward,
            'episode_end_fail_reward': self.episode_end_fail_reward,
            'sign_run_reward': self.sign_run_reward,
            'not_moving_reward': self.not_moving_reward,
            'out_of_road_reward': self.out_of_road_reward,
            'steering_reward': self.steering_reward,
            'speeding_reward': self.speeding_reward,
            'collision_reward': self.collision_reward,
            'reached_wp_reward': self.reached_wp_reward,
            'speed': self.get_velocity(),
            'throttle': self.vehicle_control.throttle,
            'brake': self.vehicle_control.brake,
            'steer': self.vehicle_control.steer,
        }

    def get_state(self):
        return self.sensor_interface.get_data()

    def episode_end(self) -> Tuple[bool, bool]:
        '''Check if the episode should end

        :return: (terminated, truncated)
        '''
        if self.stopped_count * self.fixed_delta_seconds > self.stopped_termination_seconds:
            return (True, False)
        if self.out_of_lane_count * self.fixed_delta_seconds > self.out_of_lane_termination_seconds:
            return (True, False)
        if not self.episode_alive: return (True, False)
        if len(self.collision_detector.data) > 0: return (True, False)
        if self.termination_on_run:
            for test in self.tests[:2]:
                if test.test_status == 'FAILURE':
                    return (True, False)

        return (False, False)

    def init_tests(self):
        '''Initializing all the tests and add them to the list.'''
        # this is used mainly to update the state at each step
        self.tests = []
        self.tests.append(RunningRedLightTest(self.vehicle))
        self.tests.append(RunningStopTest(self.vehicle))

        '''
        # check with ._in_lane
        self.wrong_lane_test = WrongLaneTest(self.vehicle)
        self.tests.append(self.wrong_lane_test)

        # check with ._outside_lane_active and ._wrong_lane_active
        route = [(a0.location, a1) for a0, a1 in self.world_handler.vehicle_route]
        self.outside_route_lane_test = OutsideRouteLanesTest(self.vehicle, route=route)
        self.tests.append(self.outside_route_lane_test)

        # check with ._onsidewalk_active and ._outside_lane_active
        self.on_sidewalk_test = OnSidewalkTest(self.vehicle)
        self.tests.append(self.on_sidewalk_test)

        # check with ._offroad
        self.offroad_test = OffRoadTest(self.vehicle)
        self.tests.append(self.offroad_test)
        '''

    def get_velocity(self) -> float:
        '''Returns the velocity of the vehicle in km/h'''
        v = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

    def check_out_of_road(self) -> bool:
        # get the 4 corners of the vehicle and then the corresponding waypoints
        # based on OnSidewalkTest class from srunner/scenariomanager/scenarioatomics/atomic_criteria.py
        transform = CarlaDataProvider.get_transform(self.vehicle)
        location = transform.location
        heading_vec = transform.get_forward_vector()
        heading_vec.z = 0
        heading_vec = heading_vec / math.sqrt(math.pow(heading_vec.x, 2) + \
                math.pow(heading_vec.y, 2))
        perpendicular_vec = carla.Vector3D(-heading_vec.y, heading_vec.x, 0)

        extent = self.vehicle.bounding_box.extent
        x_boundary_vector = heading_vec * extent.x
        y_boundary_vector = perpendicular_vec * extent.y

        bbox = [
            location + carla.Location(x_boundary_vector - y_boundary_vector),
            location + carla.Location(x_boundary_vector + y_boundary_vector),
            location + carla.Location(-1 * x_boundary_vector - y_boundary_vector),
            location + carla.Location(-1 * x_boundary_vector + y_boundary_vector)]

        self.bbox_wp = [
            self.map.get_waypoint(bbox[0], lane_type=carla.LaneType.Any),
            self.map.get_waypoint(bbox[1], lane_type=carla.LaneType.Any),
            self.map.get_waypoint(bbox[2], lane_type=carla.LaneType.Any),
            self.map.get_waypoint(bbox[3], lane_type=carla.LaneType.Any)]

        if any([wp.is_junction for wp in self.bbox_wp]):
            self.in_junction = True
            return False

        for i, wp in enumerate(self.bbox_wp):
            if wp.lane_type is not carla.LaneType.Driving:
                if wp.lane_type is carla.LaneType.Shoulder:
                    current_wp = self.map.get_waypoint(location, lane_type=carla.LaneType.Any)
                    distance_vehicle_wp = location.distance(current_wp.transform.location)
                    if distance_vehicle_wp >= current_wp.lane_width / 2:
                        return True
                else:
                    return True

        return False

    def check_out_of_lane(self) -> bool:
        '''
        Run first "check_out_of_road" method!!!

        If the vehicle is in a junction we can't detect if it is out of lane,
        because there are no lanes.
        '''
        # if the vehicle is out of road don't add any reward
        if self.out_of_road:
            return False

        command = self.navigation_commad
        if command not in [RoadOption.LEFT, RoadOption.RIGHT]:
            if any([wp.is_junction for wp in self.bbox_wp]):
                return False

            # waypoint of the current plan
            plan_wp = self.map.get_waypoint(self.route_planner.vehicle_route[0][0].location)
            lane_change_left = command == RoadOption.CHANGELANELEFT
            lane_change_right = command == RoadOption.CHANGELANERIGHT
            if lane_change_left:
                self.left_change_count = 10
            elif self.left_change_count > 0:
                self.left_change_count -= 1
            if lane_change_right:
                self.right_change_count = 10
            elif self.right_change_count > 0:
                self.right_change_count -= 1

            # for the front corners
            for i, wp in enumerate(self.bbox_wp[:2]):
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
                    return True

        return False


    def destroy_sensors(self):
        '''
        Destroy all the actors-sensors that this class holds.
        In order to destroy ALL the actors in the world, use the
        `destroy_actors_all` method.
        '''
        for sensor in self.sensors_env: sensor.destroy()
        for i, _ in enumerate(self.sensors):
            if self.sensors[i] is not None:
                self.sensors[i].stop()
                self.sensors[i].destroy()
                self.sensors[i] = None
        self.sensors = []

        CarlaDataProvider.get_world().tick()

    def destroy_actors_all(self):
        '''Use get_actors method to get the actors on the world'''
        actors = CarlaDataProvider.get_world().get_actors()
        actors = [x for x in actors if not x.type_id.startswith('traffic')]
        actors = [x for x in actors if not x.type_id.startswith('spectator')]

        batch = [carla.command.DestroyActor(x) for x in actors]
        # NOTE: debug, need to remove probably due to for loop
        for x in actors: logger.debug(f'Destroying sensor: {x}')

        CarlaDataProvider.get_client().apply_batch(batch)
        CarlaDataProvider.get_world().tick()

    def init_env_sensors(self):
        # camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.1))
        if self.render_rgb_camera_flag:
            # initialize Sensor class for the sensors
            camera_bp = CarlaDataProvider._blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '600')
            camera_bp.set_attribute('image_size_y', '200')
            camera_bp.set_attribute('fov', '120')

            self.rgb_camera = Sensor(camera_bp, camera_transform)
            self.sensors_env.append(self.rgb_camera)

        # collision detector
        collision_bp = CarlaDataProvider._blueprint_library.find('sensor.other.collision')
        self.collision_detector = Sensor(collision_bp, camera_transform)
        self.sensors_env.append(self.collision_detector)

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def create_observation_space(self):
        obs_space = {}
        for sensor in self.sensors_config:
            if sensor['type'] == 'sensor.camera.rgb':
                space = gym.spaces.Box(
                        low=0,
                        high=255,
                        shape=(sensor['height'], sensor['width']),
                        dtype=np.uint8,
                        )
            elif sensor['type'] == 'sensor.other.gnss':
                space = gym.spaces.Box(
                        low=float("-inf"),
                        high=float("inf"),
                        shape=(3,),
                        dtype=np.float64,
                        )
            elif sensor['type'] == 'sensor.other.imu':
                space = gym.spaces.Box(
                        low=float("-inf"),
                        high=float("inf"),
                        shape=(7,),
                        dtype=np.float64,
                        )
            elif sensor['type'] == 'sensor.speedometer':
                 space = gym.spaces.Dict({
                    'speed': gym.spaces.Box(
                        low=float("-inf"),
                        high=float("inf"),
                        shape=(1,),
                        dtype=np.float64,
                        )
                    })
            else:
                logger.warning(f"{sensor['type']} not implemented for the observation space")
                continue
            obs_space[sensor['id']] = gym.spaces.Tuple((
                gym.spaces.Box(low=0, high=float("inf"), shape=(1,), dtype=np.int64),
                space,
                ))
        return gym.spaces.Dict(obs_space)

    def reset_env_sensors(self):
        world = CarlaDataProvider.get_world()
        if self.render_rgb_camera_flag:
            self.rgb_camera.reset(
                    self.vehicle,
                    world,
                    extract_rgb_data,
                    )
        self.collision_detector.reset(
                self.vehicle,
                world,
                lambda event: event.other_actor.type_id, # blueprint
                )

    def render_rgb_camera(self):
        cv2.imshow('rgb camera', self.rgb_camera.data[-1])
        cv2.waitKey(1)

