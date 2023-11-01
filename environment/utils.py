import cv2
import os
import math
from typing import Optional

def save_image(image, name: str, path: Optional[str]=None):
    if not os.path.exists(path): os.makedirs(path)
    filepath = name if path is None else os.path.join(path, name)
    cv2.imwrite(filepath, image)

def save_env_image(env,
                   name: str='test_image.png',
                   path: Optional[str]='test_images'):
    save_image(env.rgb_camera.data[-1], name, path)


def rotate_point(point, angle: float):
    x = math.cos(angle) * point.x - math.sin(angle) * point.y
    y = math.sin(angle) * point.x + math.cos(angle) * point.y
    return x, y

def rotate_x(point, angle: float):
    return math.cos(angle) * point.x - math.sin(angle) * point.y


''' fails
from pathlib import Path
def save_image2(image, name, path=None):
    if path is None:
        filepath = name
    else:
        filepath = Path(path)
        Path(path).mkdir(parents=True, exist_ok=True)
        filepath = filepath / name
    cv2.imwrite(str(filepath), image)
'''
