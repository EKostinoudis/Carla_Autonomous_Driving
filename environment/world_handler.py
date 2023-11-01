from __future__ import annotations
import logging
import random
from itertools import cycle
import carla
from omegaconf import OmegaConf, DictConfig
from pathlib import Path

from .scenarios import ScenarioRunner
from .traffic_generator import TrafficGenerator

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

        """
        run type can be:
            "scenario": for runing scenarios
            "route": for running routes
            otherwise: free ride
        """
        self.run_type = config.get('run_type', None)
        logger.debug(f'Init: Run type: {self.run_type}')

        self.client = carla.Client(self.ip, self.port)
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
        else:
            self.num_of_vehicles = config.get('num_of_vehicles', 0)
            self.num_of_walkers = config.get('num_of_walkers', 0)

            if self.map is not None:
                # load world if given
                logger.info(f'Loading map: {self.map}')
                self.asynchronous()
                world = self.client.load_world(self.map)

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
                    yield random.choice(self.configs)
                self.configs_iter = configs_generator()
            else:
                self.configs_iter = cycle(self.configs)

    def reset(self):
        self.clean()
        GameTime.restart()

        # if we have a scenario or route run
        if self.scenario_runner:
            # pick next config
            config = next(self.configs_iter)
            logger.info(f'Scenario name: {config.name}')

            # load next scenario
            self.scenario_runner.load_scenario(config)
            self.vehicle = self.scenario_runner.vehicle
            trajectory = self.scenario_runner.trajectory
        else:
            # reload the world
            self.asynchronous()
            self.client.reload_world(True)
            self.synchronous()

            # init CarlaDataProvider
            self.world = self.client.get_world()
            CarlaDataProvider.set_world(self.world)

            # create the ego vehicle
            self.create_actor()
            trajectory = [self.vehicle.get_location(), self.destination]

            self.traffic_generator = TrafficGenerator(
                self.client,
                self.world,
                self.traffic_manager,
                self.num_of_vehicles,
                self.num_of_walkers,
                )

            # TODO: weather control???

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

        task_fail = False
        if self.scenario_runner:
            running = self.scenario_runner.tick()
            if not running:
                task_fail, _ = self.scenario_runner.scenario_manager.scenario_final_state()
        else:
            running = self.vehicle.get_location().distance(self.destination) > 5.0
        return (running, task_fail)

    def close(self):
        self.clean()
        self.asynchronous()

    def clean(self):
        if self.scenario_runner:
            self.scenario_runner.clean()
        else:
            # otherwise the CarlaDataProvider will clean the vehicle
            if self.vehicle is not None:
                self.vehicle.destroy()
                self.vehicle = None

        CarlaDataProvider.cleanup()

        if self.traffic_generator is not None:
            self.traffic_generator.destroy()
            self.traffic_generator = None

    def create_actor(self):
        '''Picks a random spawn point and spawns the vehicle'''
        spawn_points = CarlaDataProvider.get_map().get_spawn_points()
        world = CarlaDataProvider.get_world()
        vehicle_blueprint = CarlaDataProvider._blueprint_library.filter('model3')[0]
        while self.vehicle is None:
            transform = random.choice(spawn_points)

            self.vehicle = world.try_spawn_actor(
                    vehicle_blueprint,
                    transform,
                    )

            world.tick()
        CarlaDataProvider.register_actor(self.vehicle)

        self.destination = random.choice(spawn_points).location
        while self.destination == transform:
            self.destination = random.choice(spawn_points).location

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

