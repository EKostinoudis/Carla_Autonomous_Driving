import carla
import logging

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

logger = logging.getLogger(__name__)

class WorldWrapper():
    def __init__(self,
                 ip='localhost',
                 port=2000,
                 tm_port=8000,
                 map=None,
                 seed=69,
                 timeout=30.0,
                 ):
        self.ip = ip
        self.port = port
        self.tm_port = tm_port
        self.map = map
        self.seed = seed
        self.timeout = timeout

        # TODO: add try, except???
        self.client = carla.Client(self.ip, self.port)
        self.client.set_timeout(self.timeout) # secs
        client_version = self.client.get_client_version()
        server_version = self.client.get_server_version()

        if client_version != server_version:
            logger.warning('Client-Server version missmatch!')
        logger.info(f'Client version: {client_version}')
        logger.info(f'Server version: {server_version}')

        if self.map is not None:
            # load world if given
            logger.info(f'Loading map: {self.map}')
            world = self.client.load_world(self.map)

        self.world = self.client.get_world()

        # set data provicder
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)

        # logger.info(f'Map: {self.world.get_map()}')
        logger.info(f'Map: {CarlaDataProvider.get_map()}')

        # set seed
        self.world.set_pedestrians_seed(self.seed)

        self.synchronous()

        # blueprint library
        self.blueprint_library = self.world.get_blueprint_library()

        self.world.tick()

    def reset(self):
        # TODO: maybe needs more
        # logger.debug(f'World reload. Reloading the map while keeping the world settings')
        # TODO: add this???
        # self.client.reload_world(False)
        pass

    def synchronous(self):
        # set synchronous mode
        logger.debug(f'Setting synchronous mode')
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        settings.max_substeps = 15 # default 10
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
