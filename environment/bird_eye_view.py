import cv2
import numpy as np
from typing import Callable

def create_bird_eye_view_trasform(width: int, height: int) -> Callable[[np.ndarray], np.ndarray]:
    base_w, base_h = 400, 300

    src = np.float32([[0, base_h-1], [base_w-1, base_h-1], [0, 0], [base_w-1, 0]])
    # dst = np.float32([[18, base_h], [330, base_h], [152, 160], [175, 160]])
    # dst = np.float32([[14, base_h-1], [330, base_h-1], [110, 200], [220, 200]])

    bottom, up = 20, 120
    dst = np.float32([[bottom, base_h-1],
                      [base_w-bottom, base_h-1],
                      [up, 200],
                      [base_w-up, 200],
                      ])

    factor = [width / base_w, height / base_h]
    src /= factor
    dst /= factor

    transform = cv2.getPerspectiveTransform(dst, src)
    return lambda img: cv2.warpPerspective(img, transform, (width, height))

