# Based on https://github.com/zhejz/carla-roach/blob/00d6f5528296900161bcb53b62197f9d5745330c/carla_gym/core/task_actor/ego_vehicle/reward/valeo_action.py

import numpy as np
import carla

from .object_finder import loc_global_to_ref
from .object_finder import ObsManagerVehicle as OmVehicle
from .object_finder import ObsManagerPedestrian as OmPedestrian
from .hazard_actor import lbc_hazard_vehicle, lbc_hazard_walker
from .traffic_light import TrafficLightHandler

class DynamicSpeed():
    def __init__(self, ego_vehicle, maximum_speed=6.):
        self._ego_vehicle = ego_vehicle

        self.om_vehicle = OmVehicle({'max_detection_number': 10, 'distance_threshold': 15})
        self.om_pedestrian = OmPedestrian({'max_detection_number': 10, 'distance_threshold': 15})
        self.om_vehicle.attach_ego_vehicle(self._ego_vehicle)
        self.om_pedestrian.attach_ego_vehicle(self._ego_vehicle)

        self.maximum_speed = maximum_speed
        self._tl_offset = -0.8 * self._ego_vehicle.vehicle.bounding_box.extent.x

    def get(self):
        ev_transform = self._ego_vehicle.vehicle.get_transform()

        # desired_speed
        obs_vehicle = self.om_vehicle.get_observation()
        obs_pedestrian = self.om_pedestrian.get_observation()

        # all locations in ego_vehicle coordinate
        hazard_vehicle_loc = lbc_hazard_vehicle(obs_vehicle, proximity_threshold=9.5)
        hazard_ped_loc = lbc_hazard_walker(obs_pedestrian, proximity_threshold=9.5)
        light_state, light_loc, _ = TrafficLightHandler.get_light_state(self._ego_vehicle.vehicle,
                                                                        offset=self._tl_offset, dist_threshold=18.0)

        desired_spd_veh = desired_spd_ped = desired_spd_rl = desired_spd_stop = self.maximum_speed

        if hazard_vehicle_loc is not None:
            dist_veh = max(0.0, np.linalg.norm(hazard_vehicle_loc[0:2])-8.0)
            desired_spd_veh = self.maximum_speed * np.clip(dist_veh, 0.0, 5.0)/5.0

        if hazard_ped_loc is not None:
            dist_ped = max(0.0, np.linalg.norm(hazard_ped_loc[0:2])-6.0)
            desired_spd_ped = self.maximum_speed * np.clip(dist_ped, 0.0, 5.0)/5.0

        if (light_state == carla.TrafficLightState.Red or light_state == carla.TrafficLightState.Yellow):
            dist_rl = max(0.0, np.linalg.norm(light_loc[0:2])-5.0)
            desired_spd_rl = self.maximum_speed * np.clip(dist_rl, 0.0, 5.0)/5.0

        # stop sign
        stop_sign = self._ego_vehicle.criteria_stop._target_stop_sign
        stop_loc = None
        if (stop_sign is not None) and (not self._ego_vehicle.criteria_stop._stop_completed):
            trans = stop_sign.get_transform()
            tv_loc = stop_sign.trigger_volume.location
            loc_in_world = trans.transform(tv_loc)
            loc_in_ev = loc_global_to_ref(loc_in_world, ev_transform)
            stop_loc = np.array([loc_in_ev.x, loc_in_ev.y, loc_in_ev.z], dtype=np.float32)
            dist_stop = max(0.0, np.linalg.norm(stop_loc[0:2])-5.0)
            desired_spd_stop = self.maximum_speed * np.clip(dist_stop, 0.0, 5.0)/5.0

        return min(self.maximum_speed, desired_spd_veh, desired_spd_ped, desired_spd_rl, desired_spd_stop)

