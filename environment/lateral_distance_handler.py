# based on https://github.com/zhejz/carla-roach/blob/00d6f5528296900161bcb53b62197f9d5745330c/carla_gym/core/task_actor/ego_vehicle/reward/valeo_action.py
# and https://github.com/zhejz/carla-roach/blob/00d6f5528296900161bcb53b62197f9d5745330c/carla_gym/core/task_actor/common/task_vehicle.py
from collections import deque
import carla
import numpy as np

class LateralDistanceHandler:
    def __init__(self, waypoints):
        self.waypoints = deque(waypoints)

    def update(self, location):
        location.z = 0

        while len(self.waypoints) >= 2:
            wp_loc = self.waypoints[0][0].location
            wp_loc_next = self.waypoints[1][0].location
            wp_loc.z = 0
            wp_loc_next.z = 0

            dist = wp_loc.distance(location)
            next_dist = wp_loc_next.distance(location)
            if next_dist < dist:
                self.waypoints.popleft()
            else:
                break

        forward_vec = self.waypoints[1][0].location - self.waypoints[0][0].location
        if self.waypoints[1][0].location.distance(self.waypoints[0][0].location) < 0.1:
            yaw = self.waypoints[1][0].rotation.yaw
        else:
            yaw = np.rad2deg(np.arctan2(forward_vec.y, forward_vec.x))

        d_vec = location - self.waypoints[0][0].location
        np_d_vec = np.array([d_vec.x, d_vec.y], dtype=np.float32)
        wp_unit_forward = carla.Rotation(yaw=yaw).get_forward_vector()
        np_wp_unit_right = np.array([-wp_unit_forward.y, wp_unit_forward.x], dtype=np.float32)

        return np.abs(np.dot(np_wp_unit_right, np_d_vec))

