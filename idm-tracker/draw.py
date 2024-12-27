import os
import cv2
import numpy as np

from dataclasses import dataclass
from typing import Tuple


@dataclass
class BB:
    x_left: int
    y_top: int
    width: int
    height: int

def get_frame(path: str, frame: int, ext: str = "jpg") -> np.ndarray:
    raw_name = f"{frame}.{ext}"
    padded_name = (5 - len(str(frame))) * "0" + raw_name
    name = os.path.join(path, padded_name)
    img = cv2.imread(name)
    return img


def get_random_color(id: int) -> Tuple[int, int, int]:
    return (id * 1512354 % 256, id * 231245 % 256, id * 5452356 % 256)


def draw_rect(frame: np.ndarray, bb: BB, id: int, conf=None) -> None:
    color = get_random_color(id)
    try:
        cv2.rectangle(
            frame,
            (bb.x_left, bb.y_top),
            (bb.x_left + bb.width, bb.y_top + bb.height),
            color=color, thickness=1
        )
        if conf is not None:
            cv2.putText(frame, str(conf), (bb.x_left + 2, bb.y_top),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            cv2.putText(frame, str(id), (bb.x_left, bb.y_top),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    except Exception as e:
        print(id, bb)
        raise e

def draw_all_rects(img: np.ndarray, tracks: np.ndarray, conf=False):
    for t in tracks:
        id = int(t[0])
        w = int(t[3])
        h = int(t[4])
        x = int(t[1] + w // 2)
        y = int(t[2] + h // 2)
        bb = BB(x, y, w, h)
        if conf:
            draw_rect(img, bb, id, t[5])
        else:
            draw_rect(img, bb, id)
    return img
