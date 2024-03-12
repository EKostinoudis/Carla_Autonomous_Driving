# Based on https://github.com/zhejz/carla-roach/blob/00d6f5528296900161bcb53b62197f9d5745330c/carla_gym/core/obs_manager/object_finder/vehicle.py
# and https://github.com/zhejz/carla-roach/blob/00d6f5528296900161bcb53b62197f9d5745330c/carla_gym/core/obs_manager/object_finder/pedestrian.py
# and other parts of the https://github.com/zhejz/carla-roach repository

import numpy as np
import carla

class ObsManagerBase(object):
    def __init__(self):
        raise NotImplementedError

    def attach_ego_vehicle(self, parent_actor):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError

    def clean(self):
        raise NotImplementedError

def cast_angle(x):
    # cast angle to [-180, +180)
    return (x+180.0)%360.0-180.0

def carla_rot_to_mat(carla_rotation):
    """
    Transform rpy in carla.Rotation to rotation matrix in np.array

    :param carla_rotation: carla.Rotation 
    :return: np.array rotation matrix
    """
    roll = np.deg2rad(carla_rotation.roll)
    pitch = np.deg2rad(carla_rotation.pitch)
    yaw = np.deg2rad(carla_rotation.yaw)

    yaw_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    pitch_matrix = np.array([
        [np.cos(pitch), 0, -np.sin(pitch)],
        [0, 1, 0],
        [np.sin(pitch), 0, np.cos(pitch)]
    ])
    roll_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(roll), np.sin(roll)],
        [0, -np.sin(roll), np.cos(roll)]
    ])

    rotation_matrix = yaw_matrix.dot(pitch_matrix).dot(roll_matrix)
    return rotation_matrix

def rot_global_to_ref(target_rot_in_global, ref_rot_in_global):
    target_roll_in_ref = cast_angle(target_rot_in_global.roll - ref_rot_in_global.roll)
    target_pitch_in_ref = cast_angle(target_rot_in_global.pitch - ref_rot_in_global.pitch)
    target_yaw_in_ref = cast_angle(target_rot_in_global.yaw - ref_rot_in_global.yaw)

    target_rot_in_ref = carla.Rotation(roll=target_roll_in_ref, pitch=target_pitch_in_ref, yaw=target_yaw_in_ref)
    return target_rot_in_ref

def vec_global_to_ref(target_vec_in_global, ref_rot_in_global):
    """
    :param target_vec_in_global: carla.Vector3D in global coordinate (world, actor)
    :param ref_rot_in_global: carla.Rotation in global coordinate (world, actor)
    :return: carla.Vector3D in ref coordinate
    """
    R = carla_rot_to_mat(ref_rot_in_global)
    np_vec_in_global = np.array([[target_vec_in_global.x],
                                 [target_vec_in_global.y],
                                 [target_vec_in_global.z]])
    np_vec_in_ref = R.T.dot(np_vec_in_global)
    target_vec_in_ref = carla.Vector3D(x=np_vec_in_ref[0, 0], y=np_vec_in_ref[1, 0], z=np_vec_in_ref[2, 0])
    return target_vec_in_ref

def loc_global_to_ref(target_loc_in_global, ref_trans_in_global):
    """
    :param target_loc_in_global: carla.Location in global coordinate (world, actor)
    :param ref_trans_in_global: carla.Transform in global coordinate (world, actor)
    :return: carla.Location in ref coordinate
    """
    x = target_loc_in_global.x - ref_trans_in_global.location.x
    y = target_loc_in_global.y - ref_trans_in_global.location.y
    z = target_loc_in_global.z - ref_trans_in_global.location.z
    vec_in_global = carla.Vector3D(x=x, y=y, z=z)
    vec_in_ref = vec_global_to_ref(vec_in_global, ref_trans_in_global.rotation)

    target_loc_in_ref = carla.Location(x=vec_in_ref.x, y=vec_in_ref.y, z=vec_in_ref.z)
    return target_loc_in_ref

def get_loc_rot_vel_in_ev(actor_list, ev_transform):
    location, rotation, absolute_velocity = [], [], []
    for actor in actor_list:
        # location
        location_in_world = actor.get_transform().location
        location_in_ev = loc_global_to_ref(location_in_world, ev_transform)
        location.append([location_in_ev.x, location_in_ev.y, location_in_ev.z])
        # rotation
        rotation_in_world = actor.get_transform().rotation
        rotation_in_ev = rot_global_to_ref(rotation_in_world, ev_transform.rotation)
        rotation.append([rotation_in_ev.roll, rotation_in_ev.pitch, rotation_in_ev.yaw])
        # velocity
        vel_in_world = actor.get_velocity()
        vel_in_ev = vec_global_to_ref(vel_in_world, ev_transform.rotation)
        absolute_velocity.append([vel_in_ev.x, vel_in_ev.y, vel_in_ev.z])
    return location, rotation, absolute_velocity


class ObsManagerVehicle(ObsManagerBase):
    """
    Template config
    obs_configs = {
        "module": "object_finder.vehicle",
        "distance_threshold": 50.0,
        "max_detection_number": 5
    }
    """

    def __init__(self, obs_configs):
        self._max_detection_number = obs_configs['max_detection_number']
        self._distance_threshold = obs_configs['distance_threshold']

        self.vehicle = None
        self._world = None
        self._map = None

    def attach_ego_vehicle(self, vehicle, world, map):
        self.vehicle = vehicle
        self._world = world
        self._map = map

    def get_observation(self):
        ev_transform = self.vehicle.get_transform()
        ev_location = ev_transform.location
        def dist_to_ev(w): return w.get_location().distance(ev_location)

        surrounding_vehicles = []
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        for vehicle in vehicle_list:
            has_different_id = self.vehicle.id != vehicle.id
            is_within_distance = dist_to_ev(vehicle) <= self._distance_threshold
            if has_different_id and is_within_distance:
                surrounding_vehicles.append(vehicle)

        sorted_surrounding_vehicles = sorted(surrounding_vehicles, key=dist_to_ev)

        location, rotation, absolute_velocity = get_loc_rot_vel_in_ev(
            sorted_surrounding_vehicles, ev_transform)

        binary_mask, extent, road_id, lane_id = [], [], [], []
        for sv in sorted_surrounding_vehicles[:self._max_detection_number]:
            binary_mask.append(1)

            bbox_extent = sv.bounding_box.extent
            extent.append([bbox_extent.x, bbox_extent.y, bbox_extent.z])

            loc = sv.get_location()
            wp = self._map.get_waypoint(loc)
            road_id.append(wp.road_id)
            lane_id.append(wp.lane_id)

        for i in range(self._max_detection_number - len(binary_mask)):
            binary_mask.append(0)
            location.append([0, 0, 0])
            rotation.append([0, 0, 0])
            extent.append([0, 0, 0])
            absolute_velocity.append([0, 0, 0])
            road_id.append(0)
            lane_id.append(0)

        obs_dict = {
            # 'frame': self._world.get_snapshot().frame,
            'binary_mask': np.array(binary_mask, dtype=np.int8),
            'location': np.array(location, dtype=np.float32),
            'rotation': np.array(rotation, dtype=np.float32),
            'extent': np.array(extent, dtype=np.float32),
            'absolute_velocity': np.array(absolute_velocity, dtype=np.float32),
            'road_id': np.array(road_id, dtype=np.int8),
            'lane_id': np.array(lane_id, dtype=np.int8)
        }
        return obs_dict

    def clean(self):
        self.vehicle = None
        self._world = None
        self._map = None

class ObsManagerPedestrian(ObsManagerBase):
    """
    Template config
    obs_configs = {
        "module": "object_finder.pedestrian",
        "distance_threshold": 50.0,
        "max_detection_number": 5
    }
    """

    def __init__(self, obs_configs):
        self._max_detection_number = obs_configs['max_detection_number']
        self._distance_threshold = obs_configs['distance_threshold']
        self.vehicle = None
        self._world = None

    def attach_ego_vehicle(self, vehicle, world, map):
        self.vehicle = vehicle
        self._world = world
        self._map = map

    def get_observation(self):
        ev_transform = self.vehicle.get_transform()
        ev_location = ev_transform.location
        def dist_to_actor(w): return w.get_location().distance(ev_location)

        surrounding_pedestrians = []
        pedestrian_list = self._world.get_actors().filter("*walker.pedestrian*")
        for pedestrian in pedestrian_list:
            if dist_to_actor(pedestrian) <= self._distance_threshold:
                surrounding_pedestrians.append(pedestrian)

        sorted_surrounding_pedestrians = sorted(surrounding_pedestrians, key=dist_to_actor)

        location, rotation, absolute_velocity = trans_utils.get_loc_rot_vel_in_ev(
            sorted_surrounding_pedestrians, ev_transform)

        binary_mask, extent, on_sidewalk, road_id, lane_id = [], [], [], [], []
        for ped in sorted_surrounding_pedestrians[:self._max_detection_number]:
            binary_mask.append(1)

            bbox_extent = ped.bounding_box.extent
            extent.append([bbox_extent.x, bbox_extent.y, bbox_extent.z])

            loc = ped.get_location()
            wp = self._map.get_waypoint(loc, project_to_road=False, lane_type=carla.LaneType.Driving)
            if wp is None:
                on_sidewalk.append(1)
            else:
                on_sidewalk.append(0)
            wp = self._map.get_waypoint(loc)
            road_id.append(wp.road_id)
            lane_id.append(wp.lane_id)

        for i in range(self._max_detection_number - len(binary_mask)):
            binary_mask.append(0)
            location.append([0, 0, 0])
            rotation.append([0, 0, 0])
            absolute_velocity.append([0, 0, 0])
            extent.append([0, 0, 0])
            on_sidewalk.append(0)
            road_id.append(0)
            lane_id.append(0)

        obs_dict = {
            # 'frame': self._world.get_snapshot().frame,
            'binary_mask': np.array(binary_mask, dtype=np.int8),
            'location': np.array(location, dtype=np.float32),
            'rotation': np.array(rotation, dtype=np.float32),
            'absolute_velocity': np.array(absolute_velocity, dtype=np.float32),
            'extent': np.array(extent, dtype=np.float32),
            'on_sidewalk': np.array(on_sidewalk, dtype=np.int8),
            'road_id': np.array(road_id, dtype=np.int8),
            'lane_id': np.array(lane_id, dtype=np.int8)
        }

        return obs_dict

    def clean(self):
        self.vehicle = None
        self._world = None
        self._map = None

