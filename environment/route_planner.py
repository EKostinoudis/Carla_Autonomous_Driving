# This is based on the Waypointer class from:
# https://github.com/yixiao1/CILv2_multiview 

import carla
import numpy as np
import math

from srunner.tools.route_manipulation import downsample_route

from agents.navigation.local_planner import RoadOption

class RoutePlanner:
    EARTH_RADIUS_EQUA = 6378137.0

    def __init__(self, gps_route, vehicle_route, success_dist=12.) -> None:
        gps_route, vehicle_route = self.process_global_plan(gps_route, vehicle_route)
        gps_route = [([loc['lat'], loc['lon'], loc['z']], cmd) for loc, cmd in gps_route]
        self.vehicle_route = vehicle_route
        self.gps_route = gps_route
        self.success_dist = success_dist # dist from the center of the waypoint
        self.current_idx = -1

    def process_global_plan(self, global_plan_gps, global_plan_world_coord):
        ''' From autonomous agent of leaderboard '''
        ds_ids = downsample_route(global_plan_world_coord, 50)
        _global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
        _global_plan = [global_plan_gps[x] for x in ds_ids]
        return _global_plan, _global_plan_world_coord

    def step(self, gnss_data, imu_data):
        if len(self.gps_route) < 2:
            return self.gps_route[0][1]

        next_gps, _ = self.gps_route[1]
        next_vec_in_global = self.gps_to_location(next_gps) - self.gps_to_location(gnss_data)
        compass = 0.0 if np.isnan(imu_data[-1]) else imu_data[-1]
        ref_rot_in_global = carla.Rotation(yaw=np.rad2deg(compass) - 90.0)
        loc_in_ev = self.vec_global_to_ref(next_vec_in_global, ref_rot_in_global)

        self.dist_from_wp = np.sqrt(loc_in_ev.x ** 2 + loc_in_ev.y ** 2)
        self.wp_in_front = loc_in_ev.x < 0.0
        # print('Distance:', self.dist_from_wp, '| In front:', self.wp_in_front)
        if self.dist_from_wp < self.success_dist and self.wp_in_front:
            self.gps_route.pop(0)
            self.vehicle_route.pop(0)

        _, road_option = self.gps_route[0]
        return road_option

    def gps_to_location(self, gps):
        lat, lon, z = gps

        location = carla.Location(z=z)

        location.x = lon / 180.0 * (math.pi * self.EARTH_RADIUS_EQUA)

        location.y = -1.0 * math.log(math.tan((lat + 90.0) * math.pi / 360.0)) * self.EARTH_RADIUS_EQUA

        return location

    def vec_global_to_ref(self, target_vec_in_global, ref_rot_in_global):
        """
        :param target_vec_in_global: carla.Vector3D in global coordinate (world, actor)
        :param ref_rot_in_global: carla.Rotation in global coordinate (world, actor)
        :return: carla.Vector3D in ref coordinate
        """
        R = self.carla_rot_to_mat(ref_rot_in_global)
        np_vec_in_global = np.array([[target_vec_in_global.x],
                                     [target_vec_in_global.y],
                                     [target_vec_in_global.z]])
        np_vec_in_ref = R.T.dot(np_vec_in_global)
        target_vec_in_ref = carla.Vector3D(x=np_vec_in_ref[0, 0], y=np_vec_in_ref[1, 0], z=np_vec_in_ref[2, 0])
        return target_vec_in_ref

    def vec_global_to_ref(self, target_vec_in_global, ref_rot_in_global):
        """
        :param target_vec_in_global: carla.Vector3D in global coordinate (world, actor)
        :param ref_rot_in_global: carla.Rotation in global coordinate (world, actor)
        :return: carla.Vector3D in ref coordinate
        """
        R = self.carla_rot_to_mat(ref_rot_in_global)
        np_vec_in_global = np.array([[target_vec_in_global.x],
                                     [target_vec_in_global.y],
                                     [target_vec_in_global.z]])
        np_vec_in_ref = R.T.dot(np_vec_in_global)
        target_vec_in_ref = carla.Vector3D(x=np_vec_in_ref[0, 0], y=np_vec_in_ref[1, 0], z=np_vec_in_ref[2, 0])
        return target_vec_in_ref

    def carla_rot_to_mat(self, carla_rotation):
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
