import os
import cv2
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass
from typing import *


@dataclass
class BB:
    x_left: int
    y_top: int
    width: int
    height: int

def get_frame(path: str, frame: int) -> np.ndarray:
    raw_name = f"{frame}-10.jpg"
    padded_name = (5 - len(str(frame))) * "0" + raw_name
    name = os.path.join(path, padded_name)
    img = cv2.imread(name)
    return img

def get_random_color(id: int) -> Tuple[int, int, int]:
    return (id * 1512354 % 256, id * 231245 % 256, id * 5452356 % 256)

def draw_boxes_plt(df, frame, dims):
    fig = plt.figure()
    plt.xlim(0, dims[1])
    plt.ylim(0, dims[0])
    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
    for _, row in df[df["frame"]==frame].iterrows():
        id = int(row['id'])
        color = get_random_color(id)
        w = int(row['bb_width'])
        h = int(row['bb_height'])
        x = int(row['bb_left'] + w // 2)
        y = int(row['bb_top'] + h // 2)
        ax.add_patch(Rectangle((x, y), w, h, facecolor=np.array(color)/255))
    plt.show()

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

def view_frame(img: np.ndarray, df: pd.DataFrame, frame: int) -> None:
    img = img.copy()
    for _, row in df[df["frame"]==frame].iterrows():
        id = int(row['id'])
        w = int(row['bb_width'])
        h = int(row['bb_height'])
        x = int(row['bb_left'] + w // 2)
        y = int(row['bb_top'] + h // 2)
        bb = BB(x, y, w, h)
        draw_rect(img, bb, id)
    plt.imshow(img)
    plt.show()

def view_frame_arr(img: np.ndarray, tracks: np.ndarray, show=False) -> None:
    fig = plt.figure()
    img = img.copy()
    for t in tracks:
        id = int(t[0])
        w = int(t[3])
        h = int(t[4])
        x = int(t[1] + w // 2)
        y = int(t[2] + h // 2)
        bb = BB(x, y, w, h)
        draw_rect(img, bb, id)
    plt.imshow(img)
    if show:
        plt.show()

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

def load_mot(mot_file: str) -> pd.DataFrame:
    df = pd.read_csv(
        mot_file,
        names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"]
    )

    return df #type: ignore

def load_mot_road(mot_file: str) -> pd.DataFrame:
    df = pd.read_csv(
        mot_file,
        names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"]
    )

    # we are only interested in the bottom part of the file
    bottom_df = df[df["bb_top"] > 350].copy(deep=True)

    return bottom_df #type: ignore

def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    x, y, _ = flow.shape
    hsv = np.zeros((x, y, 3), dtype="uint8")
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) #type: ignore
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def view_flow(flow: np.ndarray) -> None:
    rgb = flow_to_rgb(flow)
    plt.imshow(rgb)

def view_flow_bbox(flow: np.ndarray, df: pd.DataFrame, frame: int) -> None:
    rgb = flow_to_rgb(flow)
    for _, row in df[df["frame"]==frame].iterrows():
        id = int(row['id'])
        w = int(row['bb_width'])
        h = int(row['bb_height'])
        x = int(row['bb_left'] + w // 2)
        y = int(row['bb_top'] + h // 2)
        bb = BB(x, y, w, h)
        draw_rect(rgb, bb, id)
    plt.imshow(rgb)


def play(mot: pd.DataFrame) -> None:
    for f in range(1, 1900):
        frame = get_frame("pNEUMA10/", f)

        for _, row in mot[mot["frame"]==f].iterrows():
            id = int(row['id'])
            w = int(row['bb_width'])
            h = int(row['bb_height'])
            x = int(row['bb_left'] + w // 2)
            y = int(row['bb_top'] + h // 2)
            bb = BB(x, y, w, h)
            draw_rect(frame, bb, id)
        cv2.imshow('frame', frame)
        if cv2.waitKey(0) == ord('q'):
            break

    cv2.destroyAllWindows()
