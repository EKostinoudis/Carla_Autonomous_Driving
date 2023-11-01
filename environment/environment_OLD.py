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

        self.vehicle_blueprint = self.world_wrapper.blueprint_library.filter('model3')[0]
        self.spawn_points = self.world_wrapper.world.get_map().get_spawn_points()

        # initialize Sensor class for the sensors
        camera_bp = self.world_wrapper.blueprint_library.find('sensor.camera.rgb')
        # NOTE: get attributes from config???
        camera_bp.set_attribute('image_size_x', '400')
        camera_bp.set_attribute('image_size_y', '300')
        # camera_bp.set_attribute('image_size_x', '80')
        # camera_bp.set_attribute('image_size_y', '60')
        # camera_bp.set_attribute('fov', '45')
        # camera_bp.set_attribute('sensor_tick', '0.01')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.rgb_camera = Sensor(camera_bp, camera_transform)
        self.sensors.append(self.rgb_camera)

        if 'bird_eye_view' not in self.config or self.config['bird_eye_view'] is False:
            self.bird_eye_view_transform = None
        else:
            self.bird_eye_view_transform = create_bird_eye_view_trasform(
                    400,
                    330,
                    )

        # collision detector
        collision_bp = self.world_wrapper.blueprint_library.find('sensor.other.collision')
        self.collision_detector = Sensor(collision_bp, camera_transform)
        self.sensors.append(self.collision_detector)

        # lane invasion detector
        lane_invasion_bp = self.world_wrapper.blueprint_library.find('sensor.other.lane_invasion')
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

        self.destroy_actors2()
        logger.info('Environment created')

    def reset(self, **kargs):
        # NOTE: currently we ignore a 'seed' argument but maybe should add it

        # remove all actors
        self.destroy_actors()
        # self.destroy_actors2()

        # reset the world
        self.world_wrapper.reset()
        self.map = self.world_wrapper.world.get_map() # maybe can put this in init???

        # spawn the main vehicle
        self.create_actor()

        # reset the sensors
        self.rgb_camera.reset(
                self.vehicle,
                self.world_wrapper.world,
                extract_rgb_data,
                )
        self.collision_detector.reset(
                self.vehicle,
                self.world_wrapper.world,
                lambda event: event.other_actor.type_id, # blueprint
                )
        self.lane_invasion.reset(
                self.vehicle,
                self.world_wrapper.world,
                lambda event: event.crossed_lane_markings,
                )

        # add the traffic
        self.spawn_traffic()

        # agent controller that provides us with the desired actions
        self.init_agent_controller()

        # reset controls
        self.vehicle_control.throttle = 0.
        self.vehicle_control.steer = 0.
        self.vehicle_control.brake = 0.

        # TODO: return state, maybe needs tick or a step?

    def step(self, action):
        ''' Performs one step of the simulation.
        1. Apply the given action.
        2. Calculate the reward.
        3. Check if the simulation ended.
        4. Get the latest state.
        '''

        # self.apply_action(action)
        self.auto_pilot_step()
        '''
        self.vehicle_control.steer = 0.3
        self.vehicle_control.throttle = 1.
        self.vehicle_control.brake = 0.
        self.vehicle.apply_control(self.vehicle_control)
        '''

        # TODO: add more that 1 tick???
        self.world_wrapper.world.tick()

        if self.render_rgb_camera_flag: self.render_rgb_camera()

        # TODO: remove this, only for debug
        save_env_image(self)

        # NOTE: for debug
        print('collision_detector', self.collision_detector.data)
        print('lane_invasion', self.lane_invasion.data)
        if len(self.lane_invasion.data) > 0:
            print('crossed: ', [str(d.type) for d in self.lane_invasion.data[-1]])
        self.check_lane_change()

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
        t = action['throttle']
        self.vehicle_control.steer = action['steer']
        self.vehicle_control.throttle = (t > 0) * t
        self.vehicle_control.brake = (t < 0) * -t

        # apply the action
        self.vehicle.apply_control(self.vehicle_control)

    def auto_pilot_step(self):
        '''Set the vehicle's control using the autopilot'''
        self.vehicle.apply_control(self.agent_controller.run_step())

    def get_reward(self) -> float:
        # TODO: implemention
        # vehicle out of lane
        if len(self.lane_invasion.data) > 0: self.lane_invasion.data.clear(); return -10.
        if self.check_lane_change(): return -10.


        return 0.

    def get_state(self):
        # TODO: implemention
        return None

    def episode_end(self) -> Tuple[bool, bool]:
        '''Check if the episode should end

        :return: (terminated, truncated)
        '''
        reached_goal = self.vehicle.get_location().distance(self.destination) < 5.0

        if reached_goal: return (True, False) # NOTE: maybe should return the opposite?
        if len(self.collision_detector.data) > 0: return (True, False)

        # TODO: check for max steps and maybe more???

        return (False, False)

    def init_agent_controller(self):
        '''
        Initializing an agent with a random destination in order to get the
        desired actions for the vehicle
        '''
        self.agent_controller = BasicAgent(self.vehicle)
        self.agent_controller.follow_speed_limits() # target speed to speed limit

        self.destination = random.choice(self.spawn_points).location
        self.agent_controller.set_destination(self.destination)

    def get_velocity(self) -> float:
        '''Returns the velocity of the vehicle in km/h'''
        v = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

    def create_actor(self):
        '''Picks a random spawn point and spawns the vehicle'''
        while self.vehicle is None:
            transform = random.choice(self.spawn_points)

            self.vehicle = self.world_wrapper.world.try_spawn_actor(
                    self.vehicle_blueprint,
                    transform,
                    )

            self.world_wrapper.world.tick()

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

    def check_out_of_road(self) -> bool:
        location = self.vehicle.get_location()
        waypoint = self.map.get_waypoint(location)

        # get the current waypoint from the planner
        plan_waypoint, plan_option = self.agent_controller.get_local_planner()._waypoints_queue[0]

        # if not the same sign the vehicle is on the wrong side of the road
        print('lane ids: ', plan_waypoint.lane_id, waypoint.lane_id) # TODO: remove
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
        ex = self.vehicle.bounding_box.extent.x
        if rx <= cx-ex and cx+ex <= lx: return False
        return True

    def destroy_actors(self):
        '''
        Destroy all the actors that this class holds.
        In order to destroy ALL the actors in the world, use the
        `destroy_actors2` method.
        '''
        # TODO: add all the actors
        if self.traffic_generator is not None:
            self.traffic_generator.destroy()
            self.traffic_generator = None

        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None

        for sensor in self.sensors: sensor.destroy()

        self.world_wrapper.world.tick()

    def destroy_actors2(self):
        '''Use get_actors method to get the actors on the world'''
        actors = self.world_wrapper.world.get_actors()
        actors = [x for x in actors if not x.type_id.startswith('traffic')]
        actors = [x for x in actors if not x.type_id.startswith('spectator')]

        batch = [carla.command.DestroyActor(x) for x in actors]
        # NOTE: debug, need to remove probably due to for loop
        for x in actors: logger.debug(f'Destroying sensor: {x}')

        self.world_wrapper.client.apply_batch(batch)
        self.world_wrapper.world.tick()

    def spawn_traffic(self):
        self.traffic_generator = TrafficGenerator(
                self.world_wrapper.client,
                self.world_wrapper.world,
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

