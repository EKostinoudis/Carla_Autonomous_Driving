from srunner.tools.scenario_helper import generate_target_waypoint

# this is for the follow leading vehicle scenarios
def get_next_junction(start_wp):
    end_wp = start_wp.next(1)[0]
    while not end_wp.is_junction: end_wp = end_wp.next(1)[0]
    return end_wp

def signalized_junction_turn(start_wp, left=True, repeat_turn=False):
    turn = -1 if left else 1
    wp = generate_target_waypoint(start_wp, turn)
    if repeat_turn:
        wp = generate_target_waypoint(wp, turn)
    wp = generate_target_waypoint(wp, 0)
    return get_next_junction(wp)

def signalized_junction_right_turn_7(start_wp):
    wp = generate_target_waypoint(start_wp, 1)
    wp = generate_target_waypoint(wp, 1)
    return get_next_junction(wp)

def vehicle_turn(start_wp, left=True):
    turn = -1 if left else 1
    wp = generate_target_waypoint(start_wp, turn)
    wp = generate_target_waypoint(wp, 0)
    return wp

def vehicle_turn_left_6(start_wp):
    wp = generate_target_waypoint(start_wp, -1)
    return wp.next(100)[0]

def vehicle_turn_left_7(start_wp):
    wp = start_wp.get_left_lane()
    wp = generate_target_waypoint(wp, -1)
    wp = generate_target_waypoint(wp, 0)
    return get_next_junction(wp)


def get_waypoint_from_scenario(scenario_name, ego_wp):
    if scenario_name.startswith('SignalizedJunction'):
        left = 'Left' in scenario_name
        number = int(scenario_name.split('_')[-1])

        if number == 7 and left == False: return signalized_junction_right_turn_7(ego_wp)

        repeat_turn = True if ((left and number == 2) or (not left and number == 3)) else False

        return signalized_junction_turn(ego_wp, left=left, repeat_turn=repeat_turn)
    elif scenario_name.startswith('FollowLeadingVehicle'):
        return get_next_junction(ego_wp)
    elif scenario_name.startswith('ChangeLane'):
        wp = generate_target_waypoint(ego_wp, 0)
        return generate_target_waypoint(wp, 0)
    elif scenario_name.startswith('VehicleTurning'):
        left = 'Left' in scenario_name
        number = int(scenario_name.split('_')[-1])

        if number == 6 and left:
            return vehicle_turn_left_6(ego_wp)
        if number == 7 and left:
            return vehicle_turn_left_7(ego_wp)

        return vehicle_turn(ego_wp, left=left)
    elif scenario_name.startswith('NoSignalJunctionCrossing'):
        wp = generate_target_waypoint(ego_wp, 0)
        return wp.next(80)[0]
    elif scenario_name.startswith('OppositeVehicleRunningRedLight'):
        wp = generate_target_waypoint(ego_wp, 0)
        return wp.next(80)[0]
    elif scenario_name.startswith('OtherLeadingVehicle'):
        wp = generate_target_waypoint(ego_wp, 0)
        return wp.next(400)[0]
    elif scenario_name.startswith('ManeuverOppositeDirection'):
        wp = generate_target_waypoint(ego_wp, 0)
        return wp.next(400)[0]
    elif scenario_name.startswith('CutIn'):
        wp = generate_target_waypoint(ego_wp, 0)
        return wp.next(800)[0]
    elif scenario_name.startswith('StationaryObjectCrossing'):
        wp = generate_target_waypoint(ego_wp, 0)
        return wp.next(800)[0]
    elif scenario_name.startswith('DynamicObjectCrossing'):
        wp = generate_target_waypoint(ego_wp, 0)
        return wp.next(800)[0]
    elif scenario_name.startswith('ConstructionSetupCrossing'):
        wp = generate_target_waypoint(ego_wp, 0)
        wp = generate_target_waypoint(wp, 0)
        wp = generate_target_waypoint(wp, 0)
        return wp.next(200)[0]
    elif scenario_name.startswith('ControlLoss'):
        wp = generate_target_waypoint(ego_wp, 0)
        return wp.next(800)[0]
    else:
        # return None
        raise Exception(f'Not implemented: {scenario_name}')

