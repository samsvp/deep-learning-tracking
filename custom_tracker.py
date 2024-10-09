#type: ignore
"""
Assumptions:
    - Vehicles only appear and disappear at screens edges;
        - New ids will only be issued near the screen edges or if it's highly
          unlikely that the detected vehicle is an old vehicle;
        - Exception is made during the first n frames;
    - Occlusion points may be present at an image;
    - Neural network is not entirely reliable to tell if a vehicle is or isn't present;
    - Vehicles near each other generally move with the same direction vector;
    - the velocity may fluctuate, but direction does not;
    - Small movements with 0 mean are generally noise;
"""

import os
import cv2
import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import *


@dataclass
class BB:
    x_left: int
    y_top: int
    width: int
    height: int


def get_random_color(id: int) -> Tuple[int, int, int]:
    return (id * 1512354 % 256, id * 231245 % 256, id * 5452356 % 256)


def draw_rect(frame: np.ndarray, bb: BB, id: int) -> None:
    color = get_random_color(id)
    cv2.rectangle(
        frame,
        (bb.x_left, bb.y_top),
        (bb.x_left + bb.width, bb.y_top + bb.height),
        color=color, thickness=2
    )
    cv2.putText(frame, str(id), (bb.x_left, bb.y_top),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def view_frame(path: str, df: pd.DataFrame, frame: int) -> None:
    raw_name = f"{frame}-10.jpg"
    padded_name = (5 - len(str(frame))) * "0" + raw_name
    name = os.path.join(path, padded_name)
    img = cv2.imread(name)
    for _, row in df[df["frame"]==frame].iterrows():
        id = int(row['id'])
        x = int(row['x'])
        y = int(row['y']) + 300
        w = int(row['bb_width'])
        h = int(row['bb_height'])
        bb = BB(x, y, w, h)
        draw_rect(img, bb, id)
    cv2.imshow(f"frame {frame}", img)
    cv2.waitKey(0)


def idm_predictor():
    """
    The predictor calculates the possible next position of the vehicle using
    the intelligent driver model
    """

def predictor():
    """
    Predicts the next position of the vehicle given it velocity vector and the world velocity.
    """


def create_world(df: pd.DataFrame) -> np.ndarray:
    """
    Creates an empty world. The world has a memory of the velocity vector of all cars
    that passed through its points and updates its values with new car data .
    """
    max_x = int(df["x"].max()) + 10
    max_y = int(df["y"].max()) + 10
    return np.zeros((max_x, max_y, 2))


def init_tracker(df: pd.DataFrame, first_n: int = 10) -> pd.DataFrame:
    """
    Initializes the tracker algorithm. It will generate new ids to the `first_n`
    vehicles. Afterwards, it will only try to generate ids to cars coming from the screen edges.
    """
    # assign new ids
    last_id = 0
    vehicles = df[df["frame"] == 1]
    new_df = pd.DataFrame({})
    for _, row in vehicles.iterrows():
        last_id += 1
        row["id"] = last_id
        new_df = pd.concat([new_df, row], axis=1, ignore_index=True)
    return new_df.T


def get_center(df: pd.DataFrame) -> pd.DataFrame:
    """Gets the center of the car bounding box"""
    xs = df["bb_left"] + df["bb_width"] / 2
    ys = df["bb_top"] + df["bb_height"] / 2
    my_df = pd.DataFrame({
        "frame": df["frame"],
        "x": xs,
        "y": ys,
        "bb_width": df["bb_width"],
        "bb_height": df["bb_height"],
        "id": df["id"],
        "conf": df["conf"],
    })
    return my_df


def load_mot(mot_file: str) -> pd.DataFrame:
    df = pd.read_csv(
        mot_file,
        names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"]
    )

    # we are only interested in the bottom part of the file
    bottom_df = df[df["bb_top"] > 300].copy(deep=True)
    bottom_df["bb_top"] -= 300

    return bottom_df

preds = load_mot("mots/yolo-tiny/cars-tiny-10.mot")
true_values = load_mot("10_0900_0930_D10_RM_mot.txt")

p = get_center(preds)
v = init_tracker(p)
print(v)
view_frame("pNEUMA10/", v, 1)

