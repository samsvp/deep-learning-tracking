#type: ignore
#%%
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import *


@dataclass
class BB:
    x_left: int
    y_top: int
    width: int
    height: int

def get_frame(path: str, frame: int, blur=True) -> np.ndarray:
    raw_name = f"{frame}-10.jpg"
    padded_name = (5 - len(str(frame))) * "0" + raw_name
    name = os.path.join(path, padded_name)
    img = cv2.imread(name)
    if blur:
        img = cv2.blur(img, (5, 5))
    return img

def load_mot(mot_file: str) -> pd.DataFrame:
    df = pd.read_csv(
        mot_file,
        names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"]
    )

    # we are only interested in the bottom part of the file
    # bottom_df = df[df["bb_top"] > 300].copy(deep=True)
    # bottom_df["bb_top"] -= 300

    return df


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

def view_frame(img: str, df: pd.DataFrame, frame: int) -> None:
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

def view_flow(flow: np.ndarray) -> None:
    x, y, _ = flow.shape
    hsv = np.zeros((x, y, 3), dtype="uint8")
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    print(hsv.shape, mag.shape)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    plt.imshow(rgb)

def calc_optical_flow(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow 

def find_leader(df: pd.DataFrame, frame: int, thresh: float = 15.0) -> dict:
    # for now, we pretend that the leader moves from left
    # to right
    p = df[(df["frame"] == frame) & (df["bb_top"] > 300)]
    p = p.sort_values('bb_left')
    leaders = {}
    for i, (_, row) in enumerate(p.iterrows()):
        next = p.iloc[i+1:]
        if len(next) == 0:
            break
        diff = abs(row["bb_top"] - next["bb_top"])
        next = next[diff < thresh]
        if len(next) == 0:
            continue
        leader = next.iloc[0]
        leaders[int(row['id'])] = int(leader['id'])
    return leaders


#%%
gt_mot = load_mot("10_0900_0930_D10_RM_mot.txt")
p_mot = load_mot("sort/output/pNEUMA10_8-tiny.txt")

frame1 = get_frame("pNEUMA10/", 1)
frame2 = get_frame("pNEUMA10/", 2)
frame3 = get_frame("pNEUMA10/", 3)
frame4 = get_frame("pNEUMA10/", 4)


# %%
view_frame(frame1, p_mot, 1)

# %%
view_frame(frame4, p_mot, 4)

# %%
# all areas are in the form bb_left, bb_top, width, height 
# spawn_areas are the areas new vehicles may appear
# sink_areas are the areas where vehicles will disappear
spawn_areas = [[0, 500, 100, 25], [400, 500, 50, 25]]
sink_areas = [[500, 500, 50, 25], [950, 475, 74, 50]]

# movement area where cars go from left to right
move_area = [[200, 370, 680, 110]]

#%%
for f in range(1, 1900):
    frame = get_frame("pNEUMA10/", f)

    for _, row in p_mot[p_mot["frame"]==f].iterrows():
        id = int(row['id'])
        w = int(row['bb_width'])
        h = int(row['bb_height'])
        x = int(row['bb_left'] + w // 2)
        y = int(row['bb_top'] + h // 2)
        bb = BB(x, y, w, h)
        draw_rect(frame, bb, id)
    cv2.imshow('frame', frame)
    if cv2.waitKey(100) == ord('q'):
        break

cv2.destroyAllWindows()

# %%
