import gymnasium as gym
import random
import numpy as np
import carla
import cv2
import logging
import math
from typing import Tuple

from .world_wrapper import WorldWrapper
from .traffic_generator import TrafficGenerator
from .sensor import *
from .bird_eye_view import create_bird_eye_view_trasform
from .utils import rotate_x, save_env_image
from .scenarios import ScenarioRunner
from .sensor_interface import SensorInterface
from .create_scenario_waypoints import get_waypoint_from_scenario

from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (RunningStopTest,
                                                                     RunningRedLightTest,
                                                                     WrongLaneTest,
                                                                     OutsideRouteLanesTest,
                                                                     OnSidewalkTest,
                                                                     OffRoadTest,
                                                                     )


try:
    from agents.navigation.basic_agent import BasicAgent
except ModuleNotFoundError as e:
    print("Can't import agents module.")
    print("Try run 'source set_env.sh' in order to set the PYTHONPATH environment variable.")
    print("Also, set the proper path for the CARLAROOT variable in the set_env.sh file.")
    print("")
    raise e

logger = logging.getLogger(__name__)

''' config attributes
'ip': str
'port' int
'tm_port' int
'map': str
'seed' int
'timeout' int
'num_of_vehicles' int
'num_of_walkers' int

# optional
'render_rgb_camera': bool
'bird_eye_view': bool
'''
class Environment(gym.Env):
    # TODO: create default values for the config???
    def __init__(self, config: dict):
        self.config = config

        # create the world
        logger.debug('Initializing world wrapper')
        self.world_wrapper = WorldWrapper(
                self.config['ip'],
                self.config['port'],
                self.config['tm_port'],
                self.config['map'],
                self.config['seed'],
                self.config['timeout'],
                )
        logger.debug('World wrapper initialization done')

        # set the seed
        random.seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        # TODO: add torch???

        self.render_rgb_camera_flag = \
                False if 'render_rgb_camera' not in self.config else self.config['render_rgb_camera']

        self.traffic_generator = None
        self.vehicle = None
        self.agent_controller = None
        self.sensors = []
        self.sensors2 = [] # rename this???

        self.spawn_points = CarlaDataProvider.get_map().get_spawn_points()

        # initialize Sensor class for the sensors
        camera_bp = CarlaDataProvider._blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '400')
        camera_bp.set_attribute('image_size_y', '300')
        # camera_bp.set_attribute('fov', '45')

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.rgb_camera = Sensor(camera_bp, camera_transform)
        self.sensors.append(self.rgb_camera)

        if 'bird_eye_view' not in self.config or self.config['bird_eye_view'] is False:
            self.bird_eye_view_transform = None
        else:
            self.bird_eye_view_transform = create_bird_eye_view_trasform(400, 300)

        # collision detector
        collision_bp = CarlaDataProvider._blueprint_library.find('sensor.other.collision')
        self.collision_detector = Sensor(collision_bp, camera_transform)
        self.sensors.append(self.collision_detector)

        # lane invasion detector
        lane_invasion_bp = CarlaDataProvider._blueprint_library.find('sensor.other.lane_invasion')
        self.lane_invasion = Sensor(lane_invasion_bp, camera_transform)
        self.sensors.append(self.lane_invasion)

        # for vehicle control
        self.vehicle_control = carla.VehicleControl()

        # steer, throttle-brake
        '''
        self.action_space = gym.spaces.Box(
                low=[-1., -1.],
                high=[1, 1],
                dtype=np.float32,
                seed=self.config['seed'],
                )
        '''
        self.action_space = gym.spaces.Dict({
            'steer': gym.spaces.Box(low=-1, high=1, dtype=np.float32),
            'throttle': gym.spaces.Box(low=-1, high=1, dtype=np.float32),
            }, seed=self.config['seed'])

        # TODO: define observation space
        # action space has steer and throttle?
        # self.observation_space =

        # TODO: maybe specify the reward_range
        # self.reward_range = (min reward, max reward)

        # TODO: move in other place
        # self.scenario_name = 'FollowLeadingVehicle_1'
        '''
        self.scenario_name = 'OppositeVehicleRunningRedLight_1'
        self.scenario = ScenarioRunner(scenario=self.scenario_name)
        '''
        route = 'srunner/data/routes_training.xml'
        scenario_file = 'srunner/data/all_towns_traffic_scenarios.json'
        self.scenario = ScenarioRunner(route=route,
                                  scenario_file=scenario_file,
                                  single_route=0,
                                  )


        self.destroy_actors_all()
        logger.info('Environment created')

    def reset(self, **kargs):
        # NOTE: currently we ignore a 'seed' argument but maybe should add it

        # remove all actors
        self.destroy_actors()

        # reset GameTime
        GameTime.restart()

        # clean data provider before the new scenario
        CarlaDataProvider.cleanup()

        # get the config and load the scenario
        configs = self.scenario.create_configs()
        self.scenario.load_scenario(configs[0])

        # reset the world
        # self.world_wrapper.reset()
        self.map = CarlaDataProvider.get_map()

        # initialize the vehicle
        self.init_vehicle()

        # reset the sensors
        world = CarlaDataProvider.get_world()
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
        self.lane_invasion.reset(
                self.vehicle,
                world,
                lambda event: event.crossed_lane_markings,
                )

        self.sensor_interface = SensorInterface()
        self.sensors2 = setup_sensors(self.config['sensors'],
                                      self.vehicle,
                                      self.sensor_interface)

        self.init_tests()

        # add the traffic
        # self.spawn_traffic()

        # agent controller that provides us with the desired actions
        self.init_agent_controller()

        # reset controls
        self.vehicle_control.throttle = 0.
        self.vehicle_control.steer = 0.
        self.vehicle_control.brake = 0.

        # TODO: remove this??
        world.tick()

        # TODO: return state, maybe needs tick or a step?
        return self.step([0., 0., 0.])

    def step(self, action):
        ''' Performs one step of the simulation.
        1. Apply the given action.
        2. Calculate the reward.
        3. Check if the simulation ended.
        4. Get the new state.
        '''
        # update GameTime
        world = CarlaDataProvider.get_world()
        GameTime.on_carla_tick(world.get_snapshot().timestamp)

        # self.apply_action(action)
        self.auto_pilot_step()
        '''
        self.vehicle_control.steer = -0.3
        self.vehicle_control.throttle = 0.3
        self.vehicle_control.brake = 0.
        self.vehicle.apply_control(self.vehicle_control)
        '''

        # TODO: add more that 1 tick???
        world.tick()

        # update scenario
        self.scenario_running = self.scenario.tick()
        print('scenario_running: ', self.scenario_running)


        if self.render_rgb_camera_flag: self.render_rgb_camera()

        # TODO: remove this, only for debug
        # save_env_image(self)

        # update the state of the tests
        for test in self.tests: _ = test.update()

        if not self.scenario_running:
            fail, result = self.scenario.scenario_manager.scenario_final_state()
            print('End scenario:', fail, result)

        for test in self.tests[:2]:
            print(test.name, test.test_status)
            if test.test_status == 'FAILURE':
                test.test_status = 'RESET'

        print('WrongLaneTest.in_lane:', self.wrong_lane_test._in_lane)
        print('OutsideRouteLanesTest._outside_lane_active:', self.outside_route_lane_test._outside_lane_active)
        print('OutsideRouteLanesTest._wrong_lane_active:', self.outside_route_lane_test._wrong_lane_active)
        print('OnSidewalkTest._onsidewalk_active:', self.on_sidewalk_test._onsidewalk_active)
        print('OnSidewalkTest._outside_lane_active:', self.on_sidewalk_test._outside_lane_active)
        print('OffRoadTest._offroad', self.offroad_test._offroad)

        # NOTE: for debug
        print('collision_detector', self.collision_detector.data)
        print('lane_invasion', self.lane_invasion.data)
        if len(self.lane_invasion.data) > 0:
            print('crossed: ', [str(d.type) for d in self.lane_invasion.data[-1]])

        print('Out of road: ', self.check_out_of_road())

        logger.debug(f'Velocity: {self.get_velocity():6.02f} '
                     f'Speed limit: {self.vehicle.get_speed_limit():6.02f}')


        # calculate the reward
        reward = self.get_reward()

        # get the terminated and truncated values
        terminated, truncated = self.episode_end()

        new_state = self.get_state()

        # returns (next state, reward, terminated, truncated, info)
        return new_state, reward, terminated, truncated, None

    def close(self): pass # currently empty
    def render(self): pass # currently empty

    def apply_action(self, action):
        self.vehicle_control.throttle = action[0]
        self.vehicle_control.brake = action[1]
        self.vehicle_control.steer = action[2]

        # apply the action
        self.vehicle.apply_control(self.vehicle_control)

    def auto_pilot_step(self):
        '''Set the vehicle's control using the autopilot'''
        self.vehicle.apply_control(self.agent_controller.run_step())

    def get_reward(self) -> float:
        # TODO: implemention
        # vehicle out of lane
        if len(self.lane_invasion.data) > 0: self.lane_invasion.data.clear(); return -10.
        if self.check_out_of_road(): return -10.

        return 0.

    def get_state(self):
        return self.sensor_interface.get_data()

    def episode_end(self) -> Tuple[bool, bool]:
        '''Check if the episode should end

        :return: (terminated, truncated)
        '''
        # TODO: remove this???
        if self.scenario is None:
            reached_goal = self.vehicle.get_location().distance(self.destination) < 5.0
            if reached_goal: return (True, False) # NOTE: maybe should return the opposite?

        if not self.scenario_running: return (True, False)
        if len(self.collision_detector.data) > 0: return (True, False)

        # TODO: check for max steps and maybe more???

        return (False, False)

    def set_destination(self):
        if self.scenario is not None:
            location = self.scenario.ego_transform.location
            vehicle_wp = self.map.get_waypoint(location)
            self.destination = get_waypoint_from_scenario(
                    self.scenario_name,
                    vehicle_wp,
                    ).transform.location
        else:
            self.destination = random.choice(self.spawn_points).location

    def init_agent_controller(self):
        '''
        Initializing an agent with a random destination in order to get the
        desired actions for the vehicle
        '''
        self.agent_controller = BasicAgent(self.vehicle)
        self.agent_controller.follow_speed_limits() # target speed to speed limit

        if self.scenario is not None:
            # change transform to waypoint
            route = [(self.map.get_waypoint(a0.location), a1) for a0, a1 in self.scenario.vehicle_route]
            self.agent_controller.set_global_plan(
                    route,
                    clean_queue=False)
        else:
            self.set_destination() # make sure to call this
            self.agent_controller.set_destination(self.destination)

    def init_tests(self):
        '''Initializing all the tests and add them to the list.'''
        # check with ._in_lane
        self.wrong_lane_test = WrongLaneTest(self.vehicle)

        # check with ._outside_lane_active and ._wrong_lane_active
        route = [(a0.location, a1) for a0, a1 in self.scenario.vehicle_route]
        self.outside_route_lane_test = OutsideRouteLanesTest(self.vehicle, route=route)

        # check with ._onsidewalk_active and ._outside_lane_active
        self.on_sidewalk_test = OnSidewalkTest(self.vehicle)

        # check with ._offroad
        self.offroad_test = OffRoadTest(self.vehicle)

        # this is used mainly to update the state at each step
        self.tests = []
        self.tests.append(RunningRedLightTest(self.vehicle))
        self.tests.append(RunningStopTest(self.vehicle))
        self.tests.append(self.wrong_lane_test)
        self.tests.append(self.outside_route_lane_test)
        self.tests.append(self.on_sidewalk_test)
        self.tests.append(self.offroad_test)

    def get_velocity(self) -> float:
        '''Returns the velocity of the vehicle in km/h'''
        v = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

    def init_vehicle(self):
        '''
        If we have a scenario (or route from ScenarioRunner) we get the vehicle
        from this class, else we spawn the vehicle.
        '''
        if self.scenario is not None:
            self.vehicle = self.scenario.vehicle
        else:
            self.create_actor()

    def create_actor(self):
        '''Picks a random spawn point and spawns the vehicle'''
        world = CarlaDataProvider.get_world()
        vehicle_blueprint = CarlaDataProvider._blueprint_library.filter('model3')[0]
        while self.vehicle is None:
            transform = random.choice(self.spawn_points)

            self.vehicle = world.try_spawn_actor(
                    vehicle_blueprint,
                    transform,
                    )

            world.tick()
        CarlaDataProvider.register_actor(self.vehicle)

    """
    # TODO: maybe change the name???
    def check_lane_change(self) -> bool:
        '''Check if the front wheels are on different lane'''
        # get front wheels
        wheels = self.vehicle.get_physics_control().wheels[:2]

        # lane_ids of the wheels
        lane_ids = [self.map.get_waypoint(wheel.position/100).lane_id for wheel in wheels]

        # NOTE: we olny hold the 2 front wheels
        if lane_ids[0] != lane_ids[1]: return True

        return False
    """

    def check_out_of_road(self) -> bool:
        # get the 4 corners of the vehicle and then its waypoints
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

        bbox_wp = [
            self.map.get_waypoint(bbox[0], lane_type=carla.LaneType.Any),
            self.map.get_waypoint(bbox[1], lane_type=carla.LaneType.Any),
            self.map.get_waypoint(bbox[2], lane_type=carla.LaneType.Any),
            self.map.get_waypoint(bbox[3], lane_type=carla.LaneType.Any)]


        # get the current waypoint from the planner
        plan_waypoint, plan_option = self.agent_controller.get_local_planner()._waypoints_queue[0]

        if any([wp.is_junction for wp in bbox_wp]):
            # print('Junction!!!') # TODO: remove
            return False

        for i, wp in enumerate(bbox_wp):
            # not inside the driving zone
            # print(f'side {i}: {wp.lane_type}') # TODO: remove
            if wp.lane_type not in (carla.LaneType.Driving, carla.LaneType.Parking):
                return True

            # print('     wp:', wp.lane_id, wp.road_id) # TODO: remove
            # print('plan wp:', plan_waypoint.lane_id, plan_waypoint.road_id) # TODO: remove
            if wp.lane_id != plan_waypoint.lane_id or wp.road_id != plan_waypoint.road_id:
                return True

        return False

        """
        location = self.vehicle.get_location()
        waypoint = self.map.get_waypoint(location)

        # get the current waypoint from the planner
        plan_waypoint, plan_option = self.agent_controller.get_local_planner()._waypoints_queue[0]

        # if not the same sign the vehicle is on the wrong side of the road
        print('lane ids: ', plan_waypoint.lane_id, waypoint.lane_id) # TODO: remove
        print('road ids: ', plan_waypoint.road_id, waypoint.road_id) # TODO: remove
        if plan_waypoint.lane_id * waypoint.lane_id < 0: return True

        left_lane_waypoint = waypoint.get_left_lane()
        right_lane_waypoint = waypoint.get_right_lane()

        # if the left or right road waypoints doesn't exist, just return
        if left_lane_waypoint is None or right_lane_waypoint is None: return False

        left_lane = left_lane_waypoint.transform.location
        right_lane = right_lane_waypoint.transform.location

        '''
        print('left   : ', f'{left_lane.x:9.03f}, {left_lane.y:9.03f}, {left_lane.z:9.03f}') # TODO: remove
        print('vehicle: ', f'{location.x:9.03f}, {location.y:9.03f}, {location.z:9.03f}') # TODO: remove
        print('right  : ', f'{right_lane.x:9.03f}, {right_lane.y:9.03f}, {right_lane.z:9.03f}') # TODO: remove
        '''

        # rotate the plane so the line (between the left and right lane) becomes
        # parallel to the x axis
        angle = -math.atan2(left_lane.y - right_lane.y, left_lane.x - right_lane.x)
        lx = rotate_x(left_lane, angle)
        cx = rotate_x(location, angle)
        rx = rotate_x(right_lane, angle)
        '''
        print('angle: ', angle) # TODO: remove
        print('left   : ', f'{lx:9.03f}') # TODO: remove
        print('vehicle: ', f'{cx:9.03f}') # TODO: remove
        print('right  : ', f'{rx:9.03f}') # TODO: remove
        print('vehicle extend x: ', self.vehicle.bounding_box.extent.x) # TODO: remove
        '''

        # TODO: maybe check for a lane change?
        # check if the vehicle with its boundaries is inside the road
        ex = self.vehicle.bounding_box.extent.x * 0.7
        if rx <= cx-ex and cx+ex <= lx: return False
        return True
        """

    def destroy_actors(self):
        '''
        Destroy all the actors that this class holds.
        In order to destroy ALL the actors in the world, use the
        `destroy_actors_all` method.
        '''
        # TODO: add all the actors
        if self.traffic_generator is not None:
            self.traffic_generator.destroy()
            self.traffic_generator = None

        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None

        for sensor in self.sensors: sensor.destroy()
        for sensor in self.sensors2: sensor.destroy()

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

    def spawn_traffic(self):
        self.traffic_generator = TrafficGenerator(
                CarlaDataProvider.get_client(),
                CarlaDataProvider.get_world(),
                self.world_wrapper.traffic_manager,
                self.config['num_of_vehicles'],
                self.config['num_of_walkers'],
                )

    def render_rgb_camera(self):
        cv2.imshow('rgb camera', self.rgb_camera.data[-1])
        cv2.waitKey(1)

        if self.bird_eye_view_transform is not None:
            cv2.imshow('bird eye view', self.bird_eye_view_transform(self.rgb_camera.data[-1]))
            cv2.waitKey(1)

