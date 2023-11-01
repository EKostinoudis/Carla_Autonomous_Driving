import math
import numpy as np

import carla
from agents.navigation.local_planner import RoadOption

class Waypointer:
    EARTH_RADIUS_EQUA = 6378137.0  # 6371km

    def __init__(self, world, global_plan_gps, global_route):

        self.map = world.get_map()
        # nocrash: T1 T2
        current_gnss = global_plan_gps[0][0]
        current_location = self.gps_to_location([current_gnss['lat'], current_gnss['lon'], current_gnss['z']])
        self.checkpoint = (current_location.x, current_location.y, RoadOption.LANEFOLLOW)
        self._global_plan_gps = []
        for node in global_plan_gps:
            gnss, cmd = node
            self._global_plan_gps.append(([gnss['lat'], gnss['lon'], gnss['z']], cmd))
        self.current_idx = -1

        # multi-towns: T3 to T6
        self._global_route = global_route
        self.latest_turn_cmd = None
        self.within_intersection = False
        self.next_turn_loc, self.next_turn_cmd = self._get_next_turn_loc(self._global_route)
        # print(self.latest_turn_cmd, self.next_turn_cmd)

    def tick_nc(self, gnss_data, imu_data):
        next_gps, _ = self._global_plan_gps[self.current_idx + 1]
        current_location = self.gps_to_location(gnss_data)

        next_vec_in_global = self.gps_to_location(next_gps) - self.gps_to_location(gnss_data)
        compass = 0.0 if np.isnan(imu_data[-1]) else imu_data[-1]
        ref_rot_in_global = carla.Rotation(yaw=np.rad2deg(compass) - 90.0)
        loc_in_ev = self.vec_global_to_ref(next_vec_in_global, ref_rot_in_global)

        # Fix the command given too late bug in the intersection
        self.next_idx = min(self.current_idx+1, len(self._global_plan_gps) - 2)
        _, next_road_option_0 = self._global_plan_gps[max(0, self.next_idx)]
        _, next_road_option_1 = self._global_plan_gps[self.next_idx + 1]
        if (next_road_option_0 in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]) \
                and (next_road_option_1 not in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]):
            next_road_option = next_road_option_1
        else:
            next_road_option = next_road_option_0

        if next_road_option == RoadOption.LANEFOLLOW:
            command_trigger_condition = (np.sqrt(loc_in_ev.x ** 2 + loc_in_ev.y ** 2) < 12.0 and loc_in_ev.x < 0.0)
        else:
            command_trigger_condition = (np.sqrt(loc_in_ev.x ** 2 + loc_in_ev.y ** 2) < 12.0 and loc_in_ev.x > 0.0)

        if command_trigger_condition:
            self.current_idx = min(self.current_idx+1, len(self._global_plan_gps) - 2)
            road_option = next_road_option
        else:
            self.current_idx = min(self.current_idx, len(self._global_plan_gps) - 2)
            _, road_option_0 = self._global_plan_gps[max(0, self.current_idx)]
            gps_point, road_option_1 = self._global_plan_gps[self.current_idx + 1]

            if (road_option_0 in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]) \
                    and (road_option_1 not in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]):
                road_option = road_option_1
            else:
                road_option = road_option_0

        self.checkpoint = (current_location.x, current_location.y, road_option)

        return self.checkpoint

    def tick_lb(self, gnss_data, imu_data):
        current_location = self.gps_to_location(gnss_data)
        if len(self._global_route) > 1:
            self._truncate_global_route_till_local_target(current_location)
            self._current_wp = self.map.get_waypoint(current_location, lane_type=carla.LaneType.Driving)
            self._future_wp = [self.map.get_waypoint(self._global_route[i][0].location, lane_type=carla.LaneType.Driving)
                               for i in range(min(10, len(self._global_route)))]

            compass = 0.0 if np.isnan(imu_data[-1]) else imu_data[-1]
            ref_rot_in_global = carla.Rotation(yaw=np.rad2deg(compass) - 90.0)

            next_junc_in_global = self.next_turn_loc - self.gps_to_location(gnss_data)
            loc_in_ev = self.vec_global_to_ref(next_junc_in_global, ref_rot_in_global)

            # check if the ego is 10 meters close to the junction
            if np.sqrt(loc_in_ev.x ** 2 + loc_in_ev.y ** 2) < 10.0:
                if loc_in_ev.x > 0.0:
                    if self._global_route[0][1] in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]:
                        candidate = self._global_route[0][1]
                    else:
                        candidate = self.next_turn_cmd

                else: # loc_in_ev.x < 0.0
                    # get next new junction command
                    self.latest_turn_cmd = self.next_turn_cmd
                    # print('Get into intersection: ', self.latest_turn_cmd)
                    # print('Update the next junction command')
                    self.next_turn_loc, self.next_turn_cmd = self._get_next_turn_loc(self._global_route)
                    candidate = self.latest_turn_cmd
                    self.within_intersection = True

            # within the intersection
            elif self.within_intersection:
                next_road_loc = self._global_route[0][0].location
                nextwp_in_global = next_road_loc - self.gps_to_location(gnss_data)
                nextwp_to_ego = self.vec_global_to_ref(nextwp_in_global, ref_rot_in_global)

                if self.latest_turn_cmd in [RoadOption.STRAIGHT]:
                    if abs(nextwp_to_ego.y) > 1.2:
                        candidate = RoadOption.CHANGELANELEFT if nextwp_to_ego.y < 0.0 else RoadOption.CHANGELANERIGHT
                    else:
                        candidate = self.latest_turn_cmd
                elif self.latest_turn_cmd in [RoadOption.LEFT]:
                    ra = self._global_route[0][0].rotation.yaw-(np.rad2deg(compass)-90.0)
                    if abs(nextwp_to_ego.y) > 1.2 and self.wrap_angle(ra) > 0.0:
                        candidate = RoadOption.CHANGELANERIGHT
                    else:
                        candidate = self.latest_turn_cmd
                elif self.latest_turn_cmd in [RoadOption.RIGHT]:
                    ra = self._global_route[0][0].rotation.yaw - (np.rad2deg(compass) - 90.0)
                    if abs(nextwp_to_ego.y) > 1.2 and self.wrap_angle(ra) < 0.0:
                        candidate = RoadOption.CHANGELANELEFT
                    else:
                        candidate = self.latest_turn_cmd
                # if abs(nextwp_to_ego.y) > 1.2:
                #     candidate = RoadOption.CHANGELANELEFT if nextwp_to_ego.y < 0.0 else RoadOption.CHANGELANERIGHT
                else:
                    candidate = self.latest_turn_cmd
                if not self.map.get_waypoint(next_road_loc, lane_type=carla.LaneType.Driving).is_junction:
                    self.within_intersection = False

            # check if we need to calibrate the command
            elif not self.within_intersection:
                # detect if out of the pre-plan road, if so, we calibrate the command
                road_change = [self._current_wp.lane_id != future_wp.lane_id for future_wp in self._future_wp if future_wp.road_id == self._current_wp.road_id]
                if any(road_change):
                    next_road_loc = self._global_route[road_change.index(True)][0].location
                    nextwp_in_global = next_road_loc - self.gps_to_location(gnss_data)
                    nextwp_to_ego = self.vec_global_to_ref(nextwp_in_global, ref_rot_in_global)
                    candidate = RoadOption.CHANGELANELEFT if nextwp_to_ego.y < 0.0 else RoadOption.CHANGELANERIGHT
                else:
                    candidate = self._global_route[0][1]

            else:
                raise RuntimeError()
            self.checkpoint = (current_location.x, current_location.y, candidate)
        else:
            self.checkpoint = (current_location.x, current_location.y, self.checkpoint[2])

        return self.checkpoint

    def tick_roach(self, gnss_data, imu_data):

        next_gps, _ = self._global_plan_gps[self.current_idx + 1]
        current_location = self.gps_to_location(gnss_data)

        next_vec_in_global = self.gps_to_location(next_gps) - self.gps_to_location(gnss_data)
        compass = 0.0 if np.isnan(imu_data[-1]) else imu_data[-1]
        ref_rot_in_global = carla.Rotation(yaw=np.rad2deg(compass) - 90.0)
        loc_in_ev = self.vec_global_to_ref(next_vec_in_global, ref_rot_in_global)

        if np.sqrt(loc_in_ev.x ** 2 + loc_in_ev.y ** 2) < 12.0 and loc_in_ev.x < 0.0:
            self.current_idx += 1
        self.current_idx = min(self.current_idx, len(self._global_plan_gps) - 2)

        _, road_option_0 = self._global_plan_gps[max(0, self.current_idx)]
        gps_point, road_option_1 = self._global_plan_gps[self.current_idx + 1]

        if (road_option_0 in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]) \
                and (road_option_1 not in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]):
            road_option = road_option_1
        else:
            road_option = road_option_0
        self.checkpoint = (current_location.x, current_location.y, road_option)

        obs = {'gnss': gnss_data,
               'imu': imu_data,
               'target_gps': np.array(gps_point, dtype=np.float32),
               'command': np.array([road_option.value], dtype=np.int8),
               'loc_in_ev': loc_in_ev
               }

        return obs

    def tick_mile(self, gnss_data, imu_data):

        next_gps, _ = self._global_plan_gps[self.current_idx + 1]
        current_location = self.gps_to_location(gnss_data)

        next_vec_in_global = self.gps_to_location(next_gps) - self.gps_to_location(gnss_data)
        compass = 0.0 if np.isnan(imu_data[-1]) else imu_data[-1]
        ref_rot_in_global = carla.Rotation(yaw=np.rad2deg(compass) - 90.0)
        loc_in_ev = self.vec_global_to_ref(next_vec_in_global, ref_rot_in_global)

        if np.sqrt(loc_in_ev.x ** 2 + loc_in_ev.y ** 2) < 12.0 and loc_in_ev.x < 0.0:
            self.current_idx += 1
        self.current_idx = min(self.current_idx, len(self._global_plan_gps) - 2)
        _, road_option_0 = self._global_plan_gps[max(0, self.current_idx)]
        gps_point, road_option_1 = self._global_plan_gps[self.current_idx + 1]
        # Gps waypoint after the immediate next waypoint.
        gps_point2, road_option_2 = self._global_plan_gps[min(len(self._global_plan_gps) - 1, self.current_idx + 2)]

        if (road_option_0 in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]) \
                and (road_option_1 not in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]):
            road_option = road_option_1
        else:
            road_option = road_option_0
        self.checkpoint = (current_location.x, current_location.y, road_option)

        # Handle road option for next next waypoint
        if (road_option_1 in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]) \
                and (road_option_2 not in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]):
            road_option_next = road_option_2
        else:
            road_option_next = road_option_1

        obs = {'gnss': gnss_data,
               'imu': imu_data,
               'target_gps': np.array(gps_point, dtype=np.float32),
               'command': np.array([road_option.value], dtype=np.int8),
               'target_gps_next': np.array(gps_point2, dtype=np.float32),
               'command_next': np.array([road_option_next.value], dtype=np.int8),
               }

        return obs


    def wrap_angle(self, angle):
        angle = (angle % 360 + 360) % 360
        if angle > 180:
            angle -= 360
        return angle

    def _truncate_global_route_till_local_target(self, current_location, windows_size=5):
        ev_location = current_location
        closest_idx = 0

        for i in range(len(self._global_route)-1):
            if i > windows_size:
                break

            loc0 = self._global_route[i][0].location
            loc1 = self._global_route[i+1][0].location

            wp_dir = loc1 - loc0
            wp_veh = ev_location - loc0
            dot_ve_wp = wp_veh.x * wp_dir.x + wp_veh.y * wp_dir.y + wp_veh.z * wp_dir.z

            if dot_ve_wp > 0:
                closest_idx = i+1

        if closest_idx > 0:
            self._last_route_location = carla.Location(self._global_route[0][0].location)

        self._global_route = self._global_route[closest_idx:]

    def _get_next_turn_loc(self, global_route):
        truncate_current_turn = False
        s=0
        for i, point in enumerate(global_route):
            if point[1] in [RoadOption.RIGHT, RoadOption.LEFT, RoadOption.STRAIGHT]:
                continue
            else:
                truncate_current_turn=True
                s=i
                break

        if truncate_current_turn:
            for point in global_route[s:]:
                if point[1] in [RoadOption.LANEFOLLOW, RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]:
                    continue
                else:
                    # print('Got the next junction command:', point[1], point[0].location)
                    return point[0].location, point[1]
            # print('Got to the end point:', global_route[-1][1])
            return global_route[-1][0].location, global_route[-1][1]

        # print('Got to the end point:', global_route[-1][1])
        return global_route[-1][0].location, global_route[-1][1]

    def reset_route(self, global_plan_gps, global_route):
        current_gnss = global_plan_gps[0][0]
        current_location = self.gps_to_location([current_gnss['lat'], current_gnss['lon'], current_gnss['z']])
        self.checkpoint = (current_location.x, current_location.y, RoadOption.LANEFOLLOW)
        self._global_plan_gps = []
        for node in global_plan_gps:
            gnss, cmd = node
            self._global_plan_gps.append(([gnss['lat'], gnss['lon'], gnss['z']], cmd))
        self.current_idx = -1

        self._global_route = global_route
        self.within_intersection = False
        self.next_turn_loc, self.next_turn_cmd = self._get_next_turn_loc(self._global_route)

    def gps_to_location(self, gps):
        lat, lon, z = gps
        lat = float(lat)
        lon = float(lon)
        z = float(z)

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
