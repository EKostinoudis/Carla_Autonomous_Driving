from enum import Enum
from typing import Tuple
from typing import Optional

class TrafficState(Enum):
    Zero = 0
    Light = 1
    Medium = 2
    Busy = 3

def to_traffic_state(state: str) -> Optional[TrafficState]:
    if state == 'Busy':
        return TrafficState.Busy
    elif state == 'Medium':
        return TrafficState.Medium
    elif state == 'Light':
        return TrafficState.Light
    elif state == 'Zero':
        return TrafficState.Zero
    else:
        return None

'''
Busy traffic numbers.
Town   Vehicles Pedestrians
Town01 120      120
Town02 70       70
Town03 70       70
Town04 150      80
Town05 120      120
Town06 120      80
'''

BUSY_TRAFFIC = {
    'Town01': (120, 120),
    'Town02': (70, 70),
    'Town03': (70, 70),
    'Town04': (150, 80),
    'Town05': (120, 120),
    'Town06': (120, 80),
}

def get_traffic(town: str, state: TrafficState) -> Tuple[int, int]:
    if state == TrafficState.Busy:
        div = 1
    elif state == TrafficState.Light:
        div = 2
    elif state == TrafficState.Medium:
        div = 3
    else:
        return 0, 0
    
    values = BUSY_TRAFFIC.get(town)
    if values is None:
        return 0, 0
    vehicles, pedestrians = values

    return vehicles//div, pedestrians//div
    
