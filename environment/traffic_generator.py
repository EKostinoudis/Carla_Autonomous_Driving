import carla
import random
import logging

logger = logging.getLogger(__name__)

# Based on carla/PythonAPI/examples/generate_traffic.py
class TrafficGenerator():
    def __init__(self,
                 client,
                 world,
                 traffic_manager,
                 num_of_vehicles=10,
                 num_of_walkers=0,
                 ):
        self.client = client
        self.world = world
        self.traffic_manager = traffic_manager
        self.num_of_vehicles = num_of_vehicles
        self.num_of_walkers = num_of_walkers

        self.vehicles_list = []
        self.walkers_list = []
        self.controllers_list = []
        self.controllers = []

        if self.num_of_vehicles == 0 and self.num_of_walkers == 0: return

        blueprint_library = self.world.get_blueprint_library()
        walker_blueprints = blueprint_library.filter('walker.pedestrian.*')
        vehicle_blueprints = blueprint_library.filter('vehicle.*')

        # hold only vehicles that doesn't cause accidents
        vehicle_blueprints = [x for x in vehicle_blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        vehicle_blueprints = [x for x in vehicle_blueprints if not x.id.endswith('microlino')]
        vehicle_blueprints = [x for x in vehicle_blueprints if not x.id.endswith('carlacola')]
        vehicle_blueprints = [x for x in vehicle_blueprints if not x.id.endswith('cybertruck')]
        vehicle_blueprints = [x for x in vehicle_blueprints if not x.id.endswith('t2')]
        vehicle_blueprints = [x for x in vehicle_blueprints if not x.id.endswith('sprinter')]
        vehicle_blueprints = [x for x in vehicle_blueprints if not x.id.endswith('firetruck')]
        vehicle_blueprints = [x for x in vehicle_blueprints if not x.id.endswith('ambulance')]

        # get spawn points and check if the requested vehicles can be spawned
        spawn_points = self.world.get_map().get_spawn_points()
        if len(spawn_points) - 1 < self.num_of_vehicles:
            logger.warning(f'Number of spawn points ({len(spawn_points)}) smaller than the requested vehicles ({self.num_of_vehicles})')
            logger.warning(f'Spawning only {len(spawn_points)-1} vehicles')
            self.num_of_vehicles = len(spawn_points) - 1

        # pick the spawn points
        spawn_points = random.sample(spawn_points, self.num_of_vehicles)

        spawn_batch = []
        for transform in spawn_points:
            blueprint = random.choice(vehicle_blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot
            spawn_batch.append(carla.command.SpawnActor(blueprint, transform)
                .then(carla.command.SetAutopilot(
                    carla.command.FutureActor,
                    True,
                    self.traffic_manager.get_port())))

        # run the batch in order to spawn the vehicles
        for response in self.client.apply_batch_sync(spawn_batch, True):
            if response.error:
                logger.error('Vehicle batch: ' + response.error)
            else:
                self.vehicles_list.append(response.actor_id)

        # From here we handle walkers
        # NOTE: if 0 walkers then return
        if self.num_of_walkers == 0:
            logger.debug(f'Spawned {len(self.vehicles_list)} vehicles. No walkers spawned.')
            return

        walkers_speed = []
        walkers_batch = []
        for _ in range(self.num_of_walkers):
            # choose a spawn point
            spawn_point = carla.Transform()
            spawn_point.location = self.world.get_random_location_from_navigation()

            # choose blueprint
            blueprint = random.choice(walker_blueprints)

            # disable invicible if set
            if blueprint.has_attribute('is_invincible'):
                blueprint.set_attribute('is_invincible', 'false')
            if blueprint.has_attribute('speed'):
                # walking
                walkers_speed.append(blueprint.get_attribute('speed').recommended_values[1])
                # running
                # walkers_speed.append(blueprint.get_attribute('speed').recommended_values[2])
            else:
                walkers_speed.append(0.0)

            # add command to the batch
            walkers_batch.append(carla.command.SpawnActor(blueprint, spawn_point))

        # run the spawn commands for the walkers
        walkers_speed_no_error = []
        for i, response in enumerate(self.client.apply_batch_sync(walkers_batch, True)):
            if response.error:
                logger.error('Walkers batch: ' + response.error)
            else:
                self.walkers_list.append(response.actor_id)
                walkers_speed_no_error.append(walkers_speed[i])

        # create the command for the walker controllers
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        controller_batch = [carla.command.SpawnActor(walker_controller_bp, carla.Transform(), w) \
                for w in self.walkers_list]

        # run the spawn commands for the controllers
        for response in self.client.apply_batch_sync(controller_batch, True):
            if response.error:
                logger.error('Controllers batch: ' + response.error)
            else:
                self.controllers_list.append(response.actor_id)

        self.world.tick()

        # TODO: check this
        # world.set_pedestrians_cross_factor(percentagePedestriansCrossing)

        # initialize the controllers
        self.controllers = self.world.get_actors(self.controllers_list)
        for controller, speed in zip(self.controllers, walkers_speed_no_error):
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())
            controller.set_max_speed(float(speed))

        logger.debug(f'Spawned {len(self.vehicles_list)} vehicles.')
        logger.debug(f'Spawned {len(self.walkers_list)} walkers and {len(self.controllers_list)} controllers.')

    def destroy(self):
        logger.debug('Destroying vehicle actors.')
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
        self.world.tick() # just in case
        self.vehicles_list = []

        logger.debug('Stoping the controllers and destroy the walkers and controllers.')

        # stop the controllers
        for controller in self.controllers: controller.stop()

        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walkers_list])
        self.world.tick() # just in case
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.controllers_list])
        self.world.tick() # just in case

        self.walkers_list.clear()
        self.controllers_list.clear()

