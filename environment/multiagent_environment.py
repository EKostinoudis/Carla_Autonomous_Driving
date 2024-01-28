import gymnasium as gym
import random
import numpy as np
import carla
import cv2
import logging
import math
from typing import Tuple, Optional
from omegaconf import DictConfig, OmegaConf
import torch
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.vector_env import VectorEnv

from .world_handler import WorldHandler
from .sensor import *
from .sensor_interface import SensorInterface
from .route_planner import RoutePlanner
from .carla_launcher import CarlaLauncher
from .fake_env import FakeEnv

from srunner.tools.route_manipulation import interpolate_trajectory
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    RunningStopTest,
    RunningRedLightTest,
)

from agents.navigation.local_planner import RoadOption

logger = logging.getLogger(__name__)


class MultiagentVecEnv(VectorEnv):
    '''
    Multi-agent vector environemt, we run all the agents-environments in the
    same server. We reset all at the same time. If an agent has a termination
    signal, we truncate all the agent's.
    '''
    def __init__(self,
                 config: DictConfig,
                 rllib_config: Optional[EnvContext] = None,
                 ):
        if not isinstance(config, DictConfig): config = OmegaConf.create(dict(config))
        run_type = config.get('run_type', None)
        if run_type in ['scenario', 'route']:
            raise ValueError(
                "MultiagentEnv doesn't support scenario and route runs. "
                "Only free ride."
            )

        self.num_agents = config.get('num_agents_per_server', None)
        if self.num_agents is None:
            raise ValueError("Missing 'num_agents_per_server' value on config")

        # set the seed
        seed = config.get('seed', 0)
        self.set_seed(seed)

        # throttle, break, esteer
        self.action_space = gym.spaces.Box(
                low=np.array([0., 0., -1.], dtype=np.float32),
                high=np.array([1., 1., 1.], dtype=np.float32),
                dtype=np.float32,
                seed=seed,
                )

        self.sensors_config = config.sensors

        # define observation space from sensors_config
        self.observation_space = self.create_observation_space()
        self.observation_space.seed(seed)

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

        # create world handler
        config.update({'multi_agent': True})
        self.world_handler = WorldHandler(config)

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
        self.reward_waypoint = config.get('reward_waypoint', 30.)

        self.vehicles = [None for _ in range(self.num_agents)]
        self.route_planner = [None for _ in range(self.num_agents)]
        self.sensors_envs = [[] for _ in range(self.num_agents)]
        self.sensors = [[] for _ in range(self.num_agents)]
        self.vehicle_control = [carla.VehicleControl() for _ in range(self.num_agents)]

        self.reset_list = [False for _ in range(self.num_agents)]
        self.reset_state, self.reset_info = None, None

        self.fake_subenvs = [FakeEnv(self.observation_space, self.action_space)
            for _ in range(self.num_agents)]

        # initialize the sensors that are used in the environment
        self.init_env_sensors()

        super().__init__(
            observation_space=self.observation_space,
            action_space=self.action_space,
            num_envs=self.num_agents,
        )

    def get_sub_environments(self):
        return self.fake_subenvs

    def vector_reset(self, *, seeds=None, options=None):
        if any(self.reset_list):
            raise Exception(
                f"vector_reset: At least one environment reset two times. "
                f"Reset list: {self.reset_list}"
            )
        self.reset_list = [False for _ in range(self.num_agents)]
        return self.reset_all(seeds=seeds, options=options)

    def reset_all(self, *, seeds=None, options=None):
        self.bbox_wp = [None for _ in range(self.num_agents)]
        self.in_junction = [False for _ in range(self.num_agents)]

        self.prev_steer = [0. for _ in range(self.num_agents)]
        self.stopped_count = [0 for _ in range(self.num_agents)]
        self.left_change_count = [0 for _ in range(self.num_agents)]
        self.right_change_count = [0 for _ in range(self.num_agents)]
        self.out_of_lane_count = [0 for _ in range(self.num_agents)]
        self.in_junction_count = [0 for _ in range(self.num_agents)]

        self.destroy_sensors()

        # if restart is needed for the carla server (mainly to avoid memory leaks)
        if self.carla_launcher is not None:
            self.carla_launcher.reset(restart_server=self.restart_server)
            self.restart_server = False

        # reset the world and get the new vehicle
        self.world_handler.reset()

        self.vehicles = self.world_handler.vehicle_list
        # set the route
        self.route_planner = [RoutePlanner(
            self.world_handler.gps_route_list[i],
            self.world_handler.vehicle_route_list[i],
        ) for i in range(self.num_agents)]

        # set the new map
        self.map = CarlaDataProvider.get_map()

        self.reset_env_sensors()

        # vehicle's sensors
        self.sensor_interface = [SensorInterface() for _ in range(self.num_agents)]
        self.sensors = [setup_sensors(
            self.sensors_config,
            self.vehicles[i],
            self.sensor_interface[i],
            self.use_leaderboard_setting,
        ) for i in range(self.num_agents)]

        self.init_tests()

        # reset controls
        for i in range(self.num_agents):
            self.vehicle_control[i].throttle = 0.
            self.vehicle_control[i].steer = 0.
            self.vehicle_control[i].brake = 0.

        CarlaDataProvider.get_world().tick()
        CarlaDataProvider.get_world().tick()

        actions = [[0, 0, 0] for _ in range(self.num_agents)]
        new_state, reward, terminated, truncated, info = self.vector_step(actions)
        infos = [{} for _ in range(self.num_agents)]
        return new_state, infos

    def reset_at(self, index, *, seed=None, options=None):
        if self.reset_list[index]:
            raise Exception(
                f"Environment with index: {index}, reset two times."
                f"Reset list: {self.reset_list}"
            )
        if not any(self.reset_list):
            self.reset_state, self.reset_info = self.reset_all(seeds=None, options=None)
        self.reset_list[index] = True

        if all(self.reset_list):
            self.reset_list = [False for _ in range(self.num_agents)]

        return self.reset_state[index], self.reset_info[index]

    def vector_step(self, actions):
        ''' Performs one step of the simulation.
        1. Apply the given action.
        2. Calculate the reward.
        3. Check if the simulation ended.
        4. Get the new state.
        '''
        self.reset_list = [False for _ in range(self.num_agents)]
        self.reset_state, self.reset_info = None, None

        for i in range(self.num_agents):
            self.apply_action(actions[i], i)

        # world (server) tick
        CarlaDataProvider.get_world().tick()

        # update scenario
        self.task_failed = [False for _ in range(self.num_agents)]
        _, _ = self.world_handler.step()
        self.episode_alive = [
            self.vehicles[i].get_location().distance(
                self.world_handler.destination_list[i]
            ) > 5.0 for i in range(self.num_agents)
        ]

        # get teh new state
        new_state = [self.get_state(i) for i in range(self.num_agents)]

        # get the new command for navigation
        self.navigation_commad, self.reached_wp = [], []
        for i in range(self.num_agents):
            prev_len = len(self.route_planner[i].gps_route)
            self.navigation_commad.append(self.route_planner[i].step(
                new_state[i]['GPS'][1],
                new_state[i]['IMU'][1],
            ))
            self.reached_wp.append(prev_len > len(self.route_planner[i].gps_route))

        if self.render_rgb_camera_flag: self.render_rgb_camera()

        # update the state of the tests
        for i in range(self.num_agents):
            for test in self.tests[i]: _ = test.update()

        # check if the vehicle is out of road or lane
        self.in_junction = [False for _ in range(self.num_agents)]
        self.out_of_road, self.out_of_lane = [], []
        for i in range(self.num_agents):
            self.out_of_road.append(self.check_out_of_road(i))
            self.out_of_lane.append(self.check_out_of_lane(i))
            if not self.in_junction[i]:
                self.in_junction_count[i] = 0
                if self.out_of_road[i] or self.out_of_lane[i]:
                    self.out_of_lane_count[i] += 1
                else:
                    self.out_of_lane_count[i] = 0
            else:
                if self.out_of_road[i] or self.out_of_lane[i]:
                    self.out_of_lane_count[i] += 1
                self.in_junction_count[i] += 1


        for i in range(self.num_agents):
            # update the stopped counter
            if self.get_velocity(i) < 0.5:
                self.stopped_count[i] += 1
            else:
                self.stopped_count[i] = 0

        # TODO: maybe add debug logs
        '''
        if self.debug:
            if not self.episode_alive:
                logger.debug(f'End episode. Task failed: {self.task_failed}')
            logger.debug(f'collision_detector: {self.collision_detector.data}')
            logger.debug(f'Out of road: {self.out_of_road}')
            logger.debug(f'Out of lane: {self.out_of_lane}')
            logger.debug(f'stopped count: {self.stopped_count}')
            logger.debug(f'Velocity: {self.get_velocity():6.02f} '
                         f'Speed limit: {self.vehicle.get_speed_limit():6.02f}')
        '''

        # calculate the reward
        reward, info = [], []
        terminated, truncated = [], []
        for i in range(self.num_agents):
            reward_i, info_i = self.get_reward(i)
            reward.append(reward_i)
            info.append(info_i)

            # get the terminated and truncated values
            terminated_i, truncated_i = self.episode_end(i)
            terminated.append(terminated_i)
            truncated.append(truncated_i)

        # if any agent-environment terminated, truncate all in order to reset all
        # at the same time
        if any(terminated):
            truncated = [True for _ in range(self.num_agents)]

        if not self.return_reward_info:
            info = [{} for _ in range(self.num_agents)]

        # if any vehicle reached the destination, reset the destination
        for i in range(self.num_agents):
            if not self.episode_alive[i]:
                self.update_destination(i)

        return new_state, reward, terminated, truncated, info

    def update_destination(self, idx: int):
        destination = random.choice(CarlaDataProvider.get_map().get_spawn_points())
        self.world_handler.destination_list[idx] = destination
        trajectory = [self.vehicles.get_location(), destination]
        gps_route, vehicle_route = interpolate_trajectory(
            self.world_handler.world,
            trajectory,
        )
        self.world_handler.gps_route_list[idx] = gps_route
        self.world_handler.vehicle_route_list[idx] = vehicle_route
        self.route_planner[idx] = RoutePlanner(
            self.world_handler.gps_route_list[idx],
            self.world_handler.vehicle_route_list[idx],
        )

    def render(self): pass # currently empty

    def close(self):
        self.destroy_sensors()

        self.world_handler.close()

        if self.carla_launcher is not None: self.carla_launcher.kill()

    def apply_action(self, action, idx: int):
        self.vehicle_control[idx].throttle = action[0]
        self.vehicle_control[idx].brake = action[1]
        self.vehicle_control[idx].steer = action[2]

        # apply the action
        self.vehicles[idx].apply_control(self.vehicle_control[idx])

    def get_reward(self, idx: int) -> Tuple[float, dict]:
        episode_end_success_reward = 0.
        episode_end_fail_reward = 0.
        sign_run_reward = 0.
        not_moving_reward = 0.
        out_of_road_reward = 0.
        steering_reward = 0.
        speeding_reward = 0.
        reached_wp_reward = self.reached_wp[idx] * self.reward_waypoint

        # end of scenario reward
        if not self.episode_alive[idx]:
            if self.task_failed[idx]:
                episode_end_fail_reward = self.reward_failure
            else:
                episode_end_success_reward = self.reward_success
        else:
            # stop and red light tests
            for test in self.tests[idx][:2]:
                # print(test.name, test.test_status)
                if test.test_status == 'FAILURE':
                    test.test_status = 'RESET'
                    sign_run_reward = self.reward_failure

            # vehicle too long stopped
            if self.stopped_count[idx] * self.fixed_delta_seconds > self.stopped_termination_seconds:
                not_moving_reward = self.reward_failure
            else:
                # vehicle too long out of lane
                if self.out_of_lane_count[idx] * self.fixed_delta_seconds > self.out_of_lane_termination_seconds:
                    out_of_road_reward = self.reward_wrong_lane
                else:
                    # out of road or lane
                    if self.out_of_road[idx] or self.out_of_lane[idx]:
                        out_of_road_reward = self.reward_wrong_lane

                    # steering reward (based on steer diff)
                    steering_reward = self.reward_steer * abs(self.prev_steer[idx] - self.vehicle_control[idx].steer)
                    # hold the previous steer value here, cause we only use it here
                    self.prev_steer[idx] = self.vehicle_control[idx].steer

                    # if the vehicle has speed lower than the given max, scale linearly the reward
                    # else (above the speed limit), give penalty
                    speed = self.get_velocity(idx)
                    if speed <= self.reward_max_speed:
                        speeding_reward = self.reward_speed * (speed / self.reward_max_speed)
                    else:
                        speeding_reward = -self.reward_speed

        info = {
            'episode_end_success_reward': episode_end_success_reward,
            'episode_end_fail_reward': episode_end_fail_reward,
            'sign_run_reward': sign_run_reward,
            'not_moving_reward': not_moving_reward,
            'out_of_road_reward': out_of_road_reward,
            'steering_reward': steering_reward,
            'speeding_reward': speeding_reward,
            'reached_wp_reward': reached_wp_reward,
            'speed': self.get_velocity(idx),
            'throttle': self.vehicle_control[idx].throttle,
            'brake': self.vehicle_control[idx].brake,
            'steer': self.vehicle_control[idx].steer,
        }
        total_reward = not_moving_reward + \
                       out_of_road_reward + \
                       steering_reward + \
                       speeding_reward + \
                       reached_wp_reward + \
                       episode_end_fail_reward + \
                       sign_run_reward + \
                       episode_end_success_reward
        return total_reward, info

    def get_state(self, idx: int):
        return self.sensor_interface[idx].get_data()

    def episode_end(self, idx) -> Tuple[bool, bool]:
        '''Check if the episode should end

        :return: (terminated, truncated)
        '''
        if self.stopped_count[idx] * self.fixed_delta_seconds > self.stopped_termination_seconds:
            return (True, False)
        if self.out_of_lane_count[idx] * self.fixed_delta_seconds > self.out_of_lane_termination_seconds:
            return (True, False)
        # if not self.episode_alive[idx]: return (True, False)
        if len(self.collision_detectors[idx].data) > 0: return (True, False)
        if self.termination_on_run:
            for test in self.tests[idx][:2]:
                if test.test_status == 'FAILURE':
                    return (True, False)

        return (False, False)

    def init_tests(self):
        '''Initializing all the tests and add them to the list.'''
        # this is used mainly to update the state at each step
        self.tests = [[] for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            self.tests[i].append(RunningRedLightTest(self.vehicles[i]))
            self.tests[i].append(RunningStopTest(self.vehicles[i]))

    def get_velocity(self, idx: int) -> float:
        '''Returns the velocity of the vehicle in km/h'''
        v = self.vehicles[idx].get_velocity()
        return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

    def check_out_of_road(self, idx: int) -> bool:
        # get the 4 corners of the vehicle and then the corresponding waypoints
        # based on OnSidewalkTest class from srunner/scenariomanager/scenarioatomics/atomic_criteria.py
        transform = CarlaDataProvider.get_transform(self.vehicles[idx])
        location = transform.location
        heading_vec = transform.get_forward_vector()
        heading_vec.z = 0
        heading_vec = heading_vec / math.sqrt(math.pow(heading_vec.x, 2) + \
                math.pow(heading_vec.y, 2))
        perpendicular_vec = carla.Vector3D(-heading_vec.y, heading_vec.x, 0)

        extent = self.vehicles[idx].bounding_box.extent
        x_boundary_vector = heading_vec * extent.x
        y_boundary_vector = perpendicular_vec * extent.y

        bbox = [
            location + carla.Location(x_boundary_vector - y_boundary_vector),
            location + carla.Location(x_boundary_vector + y_boundary_vector),
            location + carla.Location(-1 * x_boundary_vector - y_boundary_vector),
            location + carla.Location(-1 * x_boundary_vector + y_boundary_vector)]

        self.bbox_wp[idx] = [
            self.map.get_waypoint(bbox[0], lane_type=carla.LaneType.Any),
            self.map.get_waypoint(bbox[1], lane_type=carla.LaneType.Any),
            self.map.get_waypoint(bbox[2], lane_type=carla.LaneType.Any),
            self.map.get_waypoint(bbox[3], lane_type=carla.LaneType.Any)]

        if any([wp.is_junction for wp in self.bbox_wp[idx]]):
            self.in_junction[idx] = True
            return False

        for i, wp in enumerate(self.bbox_wp[idx]):
            if wp.lane_type is not carla.LaneType.Driving:
                if wp.lane_type is carla.LaneType.Shoulder:
                    current_wp = self.map.get_waypoint(location, lane_type=carla.LaneType.Any)
                    distance_vehicle_wp = location.distance(current_wp.transform.location)
                    if distance_vehicle_wp >= current_wp.lane_width / 2:
                        return True
                else:
                    return True

        return False

    def check_out_of_lane(self, idx: int) -> bool:
        '''
        Run first "check_out_of_road" method!!!

        If the vehicle is in a junction we can't detect if it is out of lane,
        because there are no lanes.

        idx: agent index
        '''
        # if the vehicle is out of road don't add any reward
        if self.out_of_road[idx]:
            return False

        command = self.navigation_commad[idx]
        if command not in [RoadOption.LEFT, RoadOption.RIGHT]:
            if any([wp.is_junction for wp in self.bbox_wp[idx]]):
                return False

            # waypoint of the current plan
            plan_wp = self.map.get_waypoint(self.route_planner[idx].vehicle_route[0][0].location)
            lane_change_left = command == RoadOption.CHANGELANELEFT
            lane_change_right = command == RoadOption.CHANGELANERIGHT
            if lane_change_left:
                self.left_change_count[idx] = 10
            elif self.left_change_count[idx] > 0:
                self.left_change_count[idx] -= 1
            if lane_change_right:
                self.right_change_count[idx] = 10
            elif self.right_change_count[idx] > 0:
                self.right_change_count[idx] -= 1

            # for the front corners
            for i, wp in enumerate(self.bbox_wp[idx][:2]):
                # check out of lane
                if wp.lane_id != plan_wp.lane_id:
                    if wp.lane_type is not carla.LaneType.Driving: continue
                    if lane_change_left and abs(wp.lane_id) == abs(plan_wp.lane_id)+1:
                        continue
                    if lane_change_right and abs(wp.lane_id)+1 == abs(plan_wp.lane_id):
                        continue

                    if command in [RoadOption.STRAIGHT, RoadOption.LANEFOLLOW]:
                        # if we had a left change: for the right front corner check if it still on the other lane
                        if self.left_change_count[idx] > 0 and i == 1 and abs(wp.lane_id) == abs(plan_wp.lane_id)+1:
                            continue
                        # if we had a right change: for the left front corner check if it still on the other lane
                        if self.right_change_count[idx] > 0 and i == 0 and abs(wp.lane_id)+1 == abs(plan_wp.lane_id):
                            continue
                    return True

        return False


    def destroy_sensors(self):
        '''
        Destroy all the actors-sensors that this class holds.
        In order to destroy ALL the actors in the world, use the
        `destroy_actors_all` method.
        '''
        for sensors in self.sensors_envs:
            for sensor in sensors: sensor.destroy()
        for i in range(len(self.sensors)):
            for s_i, _ in enumerate(self.sensors[i]):
                if self.sensors[i][s_i] is not None:
                    self.sensors[i][s_i].stop()
                    self.sensors[i][s_i].destroy()
                    self.sensors[i][s_i] = None
        self.sensors = [[] for _ in range(self.num_agents)]

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
        self.collision_detectors = []
        for i in range(self.num_agents):
            camera_transform = carla.Transform(carla.Location(x=1.5, z=2.1))

            # collision detector
            collision_bp = CarlaDataProvider._blueprint_library.find('sensor.other.collision')
            self.collision_detectors.append(Sensor(collision_bp, camera_transform))
            self.sensors_envs[i].append(self.collision_detectors[i])

            if i == 0 and self.render_rgb_camera_flag:
                # initialize Sensor class for the sensors
                camera_bp = CarlaDataProvider._blueprint_library.find('sensor.camera.rgb')
                camera_bp.set_attribute('image_size_x', '600')
                camera_bp.set_attribute('image_size_y', '200')
                camera_bp.set_attribute('fov', '120')

                self.rgb_camera = Sensor(camera_bp, camera_transform)
                self.sensors_envs[i].append(self.rgb_camera)

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
        for i in range(self.num_agents):
            self.collision_detectors[i].reset(
                    self.vehicles[i],
                    world,
                    lambda event: event.other_actor.type_id, # blueprint
                    )
        if self.render_rgb_camera_flag:
            self.rgb_camera.reset(
                self.vehicles[0],
                world,
                extract_rgb_data,
                )

    def render_rgb_camera(self):
        cv2.imshow('rgb camera', self.rgb_camera.data[-1])
        cv2.waitKey(1)

