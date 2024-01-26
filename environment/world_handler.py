from __future__ import annotations
import logging
import random
from itertools import cycle
import carla
from omegaconf import OmegaConf, DictConfig
from pathlib import Path

from environment.traffic_calculator import get_traffic

from .scenarios import ScenarioRunner
from .traffic_generator import TrafficGenerator
from .weather_handler import WeatherHandler
from .traffic_calculator import get_traffic, TrafficState, to_traffic_state

from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.tools.route_manipulation import interpolate_trajectory


logger = logging.getLogger(__name__)

class WorldHandler():
    def __init__(self, config: DictConfig):
        self.ip = config.get('ip', 'localhost')
        self.port = config.get('port', 2000)
        self.tm_port = config.get('tm_port', 8000)
        self.map = config.get('map', None)
        self.seed = config.get('seed', 0)
        self.timeout = config.get('timeout', 30.)
        self.fixed_delta_seconds = config.get('fixed_delta_seconds', 0.1)
        self.max_substeps = config.get('max_substeps', 15) # default value is 10

        self.vehicle = None
        self.scenario_runner = None
        self.traffic_generator = None
        self.gps_route, self.vehicle_route = None, None

        # for multi agent
        self.vehicle_list = []
        self.destination_list = []
        self.gps_route_list, self.vehicle_route_list = [], []

        """
        run type can be:
            "scenario": for runing scenarios
            "route": for running routes
            otherwise: free ride
        """
        self.run_type = config.get('run_type', None)
        self.multi_agent = config.get('multi_agent', False)
        self.num_agents = config.get('num_agents_per_server', 0)
        logger.debug(f'Init: Run type: {self.run_type}')

        self.client = carla.Client(self.ip, self.port, config.get('worker_threads', 0))
        self.client.set_timeout(self.timeout) # secs
        client_version = self.client.get_client_version()
        server_version = self.client.get_server_version()

        if client_version != server_version:
            logger.warning('Client-Server version missmatch!')
        logger.debug(f'Client version: {client_version}')
        logger.debug(f'Server version: {server_version}')

        self.world = self.client.get_world()

        # set data provicder
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)

        # init the scenario runner if the type is 'scenario' or 'route'
        if self.run_type == 'scenario':
            self.scenario_runner = ScenarioRunner(
                    client=self.client,
                    scenario=config.get('scenario_config', 'all'),
                    scenario_config=config.get('scenario_config', ''),
                    host=self.ip,
                    port=self.port,
                    timeout=self.timeout,
                    tm_port=self.tm_port,
                    tm_seed=self.seed,
                    randomize=config.get('randomize', True),
                    fixed_delta_seconds=self.fixed_delta_seconds,
                    max_substeps=self.max_substeps,
                    )
            if OmegaConf.select(config, 'scenario') is None:
                file = Path(__file__).resolve().parent / 'all_scenarios.txt'
                scenario_names = open(file, 'r').read().strip().split('\n')
                self.configs = ScenarioRunner.create_scenarios_list(scenario_names)
            else:
                self.configs = self.scenario_runner.create_configs()
        elif self.run_type == 'route':
            if OmegaConf.select(config, 'route') is None:
                logger.warning('Missing "route" argument in config. Running in free ride mode.')
                self.run_type = None
            elif OmegaConf.select(config, 'scenario_file') is None:
                logger.warning('Missing "scenario_file" argument in config. Running in free ride mode.')
                self.run_type = None

            self.scenario_runner = ScenarioRunner(
                    client=self.client,
                    route=config.route,
                    scenario_file=config.scenario_file,
                    single_route=config.get('single_route', None),
                    host=self.ip,
                    port=self.port,
                    timeout=self.timeout,
                    tm_port=self.tm_port,
                    tm_seed=self.seed,
                    randomize=config.get('randomize', True), # useless in route scenarios
                    fixed_delta_seconds=self.fixed_delta_seconds,
                    max_substeps=self.max_substeps,
                    )
            self.configs = self.scenario_runner.create_configs()
        else: # FREE RIDE
            self.training_mode = config.get('training_mode', False)
            if self.training_mode:
                def map_picker():
                    while True:
                        yield random.choice([
                            'Town01',
                            'Town02',
                            'Town03',
                            'Town04',
                            'Town06',
                        ])
                self.map_picker = map_picker()
                def traffic_state_picker():
                    while True:
                        yield random.choice([
                            TrafficState.Zero,
                            TrafficState.Light,
                            TrafficState.Medium,
                            TrafficState.Busy,
                        ])
                self.traffic_state_picker = traffic_state_picker()
            else:
                if "num_of_vehicles" in config and "num_of_walkers" in config:
                    self.num_of_vehicles = config.get('num_of_vehicles', 0)
                    self.num_of_walkers = config.get('num_of_walkers', 0)
                else:
                    # if the traffic state is given and it is valid use it,
                    # else pick a random state
                    traffic_state = config.get('traffic_state', None)
                    traffic_state = to_traffic_state(traffic_state)
                    if traffic_state is None:
                        traffic_state = random.choice(list(TrafficState))
                    self.num_of_vehicles, self.num_of_walkers = get_traffic(
                        CarlaDataProvider.get_map().name[-6:],
                        traffic_state,
                    )

                if self.map is not None:
                    # load world if given
                    logger.info(f'Loading map: {self.map}')
                    # self.asynchronous()
                    world = self.client.load_world(self.map)

            # create the weather handler
            self.weather_handler = WeatherHandler()
            self.random_weather = config.get('random_weather', False)
            self.dynamic_weather = config.get('dynamic_weather', False)

            self.world = self.client.get_world()

            # set data provicder
            CarlaDataProvider.set_world(self.world)
            logger.info(f'Map: {CarlaDataProvider.get_map()}')

            # set seed
            self.world.set_pedestrians_seed(self.seed)

            self.synchronous()
            self.world.tick()

        # if there is a scenario or route create an iterator for the configs
        if self.scenario_runner:
            if config.get('pick_random', True):
                def configs_generator():
                    while True:
                        yield random.choice(self.configs)
                self.configs_iter = configs_generator()
            else:
                self.configs_iter = cycle(self.configs)

    def reset(self):
        self.clean()
        GameTime.restart()

        # if we have a scenario or route run
        if self.scenario_runner is not None:
            # pick next config
            config = next(self.configs_iter)
            logger.info(f'Scenario name: {config.name}')

            # load next scenario
            self.scenario_runner.load_scenario(config)
            self.vehicle = self.scenario_runner.vehicle
            trajectory = self.scenario_runner.trajectory
        else:
            self.synchronous()
            if self.training_mode:
                world = self.client.load_world(next(self.map_picker))
            else:
                self.client.reload_world(True)
            self.synchronous()

            # init CarlaDataProvider
            self.world = self.client.get_world()
            CarlaDataProvider.set_world(self.world)

            if self.training_mode:
                self.num_of_vehicles, self.num_of_walkers = get_traffic(
                    CarlaDataProvider.get_map().name[-6:],
                    next(self.traffic_state_picker),
                )

            # weather control
            self.weather_handler.reset(
                random_weather=self.random_weather,
                dynamic_weather=self.dynamic_weather,
            )

            # create the ego vehicle
            self.vehicle, self.destination = self.create_actor()
            trajectory = [self.vehicle.get_location(), self.destination]

            self.vehicle_list, self.destination_list = [], []
            self.gps_route_list = []
            self.vehicle_route_list = []
            if self.multi_agent:
                self.vehicle_list.append(self.vehicle)
                self.destination_list.append(self.destination)
                trajectory_list = [trajectory]
                for _ in range(1, self.num_agents):
                    vehicle, destination = self.create_actor()
                    self.destination_list.append(self.destination)
                    self.vehicle_list.append(vehicle)
                    self.destination_list.append(destination)
                    trajectory_m = [vehicle.get_location(), destination]
                    trajectory_list.append(trajectory_m)

                for i in range(self.num_agents):
                    gps_route, vehicle_route = interpolate_trajectory(self.world, trajectory_list[i])
                    self.gps_route_list.append(gps_route)
                    self.vehicle_route_list.append(vehicle_route)

            self.traffic_generator = TrafficGenerator(
                self.client,
                self.world,
                self.traffic_manager,
                self.num_of_vehicles,
                self.num_of_walkers,
            )

        self.gps_route, self.vehicle_route = interpolate_trajectory(self.world, trajectory)

    def step(self) -> tuple[bool, bool]:
        '''
        Perform a step (not client tick)

        return: if the simulation is alive (used only for the scenarios and routes)
                and if the task failed
        '''
        # update GameTime
        world = CarlaDataProvider.get_world()
        GameTime.on_carla_tick(world.get_snapshot().timestamp)

        # update data provider
        CarlaDataProvider.on_carla_tick()

        task_fail = False
        if self.scenario_runner is not None:
            running = self.scenario_runner.tick()
            if not running:
                task_fail, _ = self.scenario_runner.scenario_manager.scenario_final_state()
        else:
            self.weather_handler.tick(self.fixed_delta_seconds)

            # end condition
            running = self.vehicle.get_location().distance(self.destination) > 5.0
        return (running, task_fail)

    def close(self):
        self.clean()
        # self.asynchronous()

        # load a small map
        # self.client.load_world('Town01_Opt')

    def clean(self):
        if self.scenario_runner is not None:
            self.scenario_runner.clean()
        else:
            if self.multi_agent:
                for vehicle in self.vehicle_list:
                    if vehicle is not None:
                        vehicle.destroy()
            else:
                if self.vehicle is not None:
                    self.vehicle.destroy()
                    self.vehicle = None

        CarlaDataProvider.cleanup()
        CarlaDataProvider.set_client(self.client)

        if self.traffic_generator is not None:
            self.traffic_generator.destroy()
            self.traffic_generator = None

    def create_actor(self):
        '''Picks a random spawn point and spawns the vehicle'''
        spawn_points = CarlaDataProvider.get_map().get_spawn_points()
        world = CarlaDataProvider.get_world()
        # vehicle_blueprint = CarlaDataProvider._blueprint_library.filter('model3')[0]
        vehicle_blueprint = CarlaDataProvider._blueprint_library.filter('vehicle.lincoln.mkz_2017')[0]
        vehicle = None
        while vehicle is None:
            transform = random.choice(spawn_points)

            vehicle = world.try_spawn_actor(
                    vehicle_blueprint,
                    transform,
                    )

            world.tick()
        CarlaDataProvider.register_actor(vehicle)

        destination = random.choice(spawn_points).location
        '''
        # don't even know why I used that, probably bug
        while self.destination == transform:
            destination = random.choice(spawn_points).location
        '''
        return vehicle, destination

    def synchronous(self):
        # set synchronous mode
        logger.debug(f'Setting synchronous mode')
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        settings.max_substeps = self.max_substeps
        self.world.apply_settings(settings)

        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(self.seed) # define TM seed for determinism

    def asynchronous(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)

        self.client.get_trafficmanager(self.tm_port).set_synchronous_mode(False)

