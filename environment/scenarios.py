import glob
import os
import sys
import importlib
import inspect
import py_trees
import carla

from .create_scenario_waypoints import get_waypoint_from_scenario

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.tools.scenario_parser import ScenarioConfigurationParser
from srunner.tools.route_parser import RouteParser
from srunner.scenarios.route_scenario import RouteScenario
# from srunner.tools.route_manipulation import interpolate_trajectory

'''
Based on ScenarioManager class in
srunner/scenariomanager/scenario_manager.py
'''
class ScenarioManager():
    def __init__(self):
        self.scenario_class = None
        self.scenario = None
        self.scenario_tree = None
        self.ego_vehicles = None
        self.other_actors = None
        self._running = False

    def load_scenario(self, scenario):
        self.scenario_class = scenario
        self.scenario = scenario.scenario
        self.scenario_tree = self.scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors

        self._running = True

    def tick_scenario(self) -> bool:
        ''' Perform a step of the scenario and return if the scenario has ended '''
        if not self._running: return False

        self.scenario_tree.tick_once()

        if self.scenario_tree.status != py_trees.common.Status.RUNNING:
            self._running = False

        return self._running

    def clean(self):
        if self.scenario is not None:
            self.scenario.terminate()

    def scenario_final_state(self):
        failure = False
        timeout = False
        result = "SUCCESS"

        if self.scenario.test_criteria is None:
            return True, result

        for criterion in self.scenario.get_criteria():
            if (not criterion.optional and
                    criterion.test_status != "SUCCESS" and
                    criterion.test_status != "ACCEPTABLE"):
                failure = True
                result = "FAILURE"
            elif criterion.test_status == "ACCEPTABLE":
                result = "ACCEPTABLE"

        if self.scenario.timeout_node.timeout and not failure:
            timeout = True
            result = "TIMEOUT"

        return failure or timeout, result

'''Based on ScenarioRunner class of scenario runner'''
class ScenarioRunner():
    def __init__(self,
                 client=None,
                 scenario='',
                 scenario_config='',
                 route='',
                 scenario_file='',
                 single_route=None,
                 host='localhost',
                 timeout=30.,
                 port=2000,
                 tm_port=8000,
                 tm_seed=0,
                 randomize=True,
                 fixed_delta_seconds=0.1,
                 max_substeps=10,
                 ):
        '''
        There are two options:
            run a scenario: must provide a `scenario`
            run a route: must provide a `route` and a `scenario_file`
        '''
        assert (scenario and not route and not scenario_file) or \
                (not scenario and route and scenario_file), \
                'Should provide a scenario or a route'
        self.is_scenario_run = True if scenario else False
        self.scenario = scenario
        self.scenario_config = scenario_config
        self.route = route
        self.scenario_file = scenario_file
        self.single_route = single_route
        self.tm_port = tm_port
        self.tm_seed = tm_seed
        self.randomize = randomize
        self.fixed_delta_seconds = fixed_delta_seconds
        self.max_substeps = max_substeps

        if client is not None:
            self.client = client
        else:
            self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)

        self.scenario_manager = ScenarioManager()

        self.ego_vehicles = []
        self.vehicle = None
        # self.gps_route, self.vehicle_route = None, None
        self.trajectory = None
        self.running_scenario = None

    def create_configs(self):
        '''Return the scenario configurations provided in the config file'''
        if self.is_scenario_run:
            return self._create_scenarios_configs()
        else:
            return self._create_route_configs()

    def _create_scenarios_configs(self):
        '''Scenario configurations'''
        return self.create_scenarios_config(self.scenario, self.scenario_config)

    def _create_route_configs(self):
        '''Route configurations'''
        single_route = str(self.single_route) if self.single_route is not None else None
        return RouteParser.parse_routes_file(self.route,
                                             self.scenario_file,
                                             single_route,
                                             )
    @staticmethod
    def create_scenarios_config(scenario, scenario_config=''):
        '''Scenario configurations'''
        return ScenarioConfigurationParser.parse_scenario_configuration(
            scenario,
            scenario_config)

    @staticmethod
    def create_scenarios_list(scenarios_names):
        return [ScenarioRunner.create_scenarios_config(scenario)[0] for scenario in scenarios_names]

    def load_scenario(self, config, agent=None):
        '''Given a config load the scenario'''
        self._load_world(config.town)

        # prepare traffic manager
        CarlaDataProvider.set_traffic_manager_port(int(self.tm_port))
        tm = self.client.get_trafficmanager(int(self.tm_port))
        tm.set_random_device_seed(int(self.tm_seed))
        tm.set_synchronous_mode(True)

        for vehicle in config.ego_vehicles:
            self.ego_vehicles.append(CarlaDataProvider.request_new_actor(
                vehicle.model,
                vehicle.transform,
                vehicle.rolename,
                color=vehicle.color,
                actor_category=vehicle.category,
                safe_blueprint=True)) # added argument
            # hold the ego vehicles spawn point
            self.ego_transform = vehicle.transform

        # pass the agent to the config (probably only usefull in routes)
        config.agent = agent

        if self.is_scenario_run:
            scenario_class = self._get_scenario_class_or_fail(config.type)
            scenario = scenario_class(self.world,
                                      self.ego_vehicles,
                                      config,
                                      self.randomize,
                                      False) # debug
            self.vehicle = self.ego_vehicles[0]
            vehicle_location = self.vehicle.get_location()
            vehicle_wp = CarlaDataProvider.get_map().get_waypoint(vehicle_location)
            end_location = get_waypoint_from_scenario(
                    config.name,
                    vehicle_wp,
                    ).transform.location
            self.trajectory = [vehicle_location, end_location]
        else:
            # route
            '''
            This class spawns the ego vehicle and sets its rolename to 'hero'.
            The spawn point is the self.route[0][0] (with + 0.5 on z axis).
            '''
            scenario = RouteScenario(self.world,
                                     config,
                                     criteria_enable=True,
                                     )
            self.ego_transform = scenario.route[0][0]
            self.vehicle = CarlaDataProvider.get_hero_actor()
            self.trajectory = config.trajectory
        # self.gps_route, self.vehicle_route = interpolate_trajectory(self.world, trajectory)

        self.scenario_manager.load_scenario(scenario)
        self.running_scenario = scenario

    def tick(self) -> bool:
        # TODO: maybe return the status of the termination???

        # update data provider
        CarlaDataProvider.on_carla_tick()
        return self.scenario_manager.tick_scenario()

    def clean(self):
        if self.running_scenario:
            self.running_scenario.remove_all_actors()
            self.running_scenario = None

        if self.scenario_manager is not None:
            self.scenario_manager.clean()

        self.ego_vehicles = []
        self.vehicle = None
        self.gps_route, self.vehicle_route = None, None

    def _set_asynchronous(self):
        self.world = self.client.get_world()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)

        self.client.get_trafficmanager(int(self.tm_port)).set_synchronous_mode(False)

    def _set_synchronous_world(self):
        self.world = self.client.get_world()

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        settings.max_substeps = self.max_substeps
        self.world.apply_settings(settings)

    def _load_world(self, town):
        """
        Load a new CARLA world and provide data to CarlaDataProvider
        """
        self._set_synchronous_world()
        self.world = self.client.load_world(town)
        self._set_synchronous_world()
        self.world.tick()

        CarlaDataProvider.set_world(self.world)
        self.world.tick()

    def _get_scenario_class_or_fail(self, scenario):
        """
        Get scenario class by scenario name
        If scenario is not supported or not found, exit script
        """

        # Path of all scenario at "srunner/scenarios" folder + the path of the additional scenario argument
        scenarios_list = glob.glob("{}/srunner/scenarios/*.py".format(os.getenv('SCENARIO_RUNNER_ROOT', "./")))
        # scenarios_list.append(self._args.additionalScenario)

        for scenario_file in scenarios_list:
            # Get their module
            module_name = os.path.basename(scenario_file).split('.')[0]
            sys.path.insert(0, os.path.dirname(scenario_file))
            scenario_module = importlib.import_module(module_name)

            # And their members of type class
            for member in inspect.getmembers(scenario_module, inspect.isclass):
                if scenario in member:
                    return member[1]

            # Remove unused Python paths
            sys.path.pop(0)

        print("Scenario '{}' not supported ... Exiting".format(scenario))
        sys.exit(-1)

