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

def calc_optical_flow(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow 

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


#%%
gt_mot = load_mot("10_0900_0930_D10_RM_mot.txt")

frame1 = get_frame("pNEUMA10/", 1)
frame2 = get_frame("pNEUMA10/", 2)
frame3 = get_frame("pNEUMA10/", 3)
frame4 = get_frame("pNEUMA10/", 4)

t_id = 40
flow = calc_optical_flow(frame2, frame3)

t = gt_mot[(gt_mot["frame"] == 2) & (gt_mot["id"] == t_id)]
x = int((t["bb_left"] + t["bb_width"] // 2).values[0])
y = int((t["bb_top"] + t["bb_height"] // 2).values[0])
print(flow[y, x, :])

t_current = gt_mot[(gt_mot["frame"] == 3) & (gt_mot["id"] == t_id)]

print(t, t_current)

#%%
for i in range(2, 10):
    frame1 = get_frame("pNEUMA10/", i - 1)
    frame2 = get_frame("pNEUMA10/", i)

    t_id = 40
    flow = calc_optical_flow(frame1, frame2)

    t = gt_mot[(gt_mot["frame"] == i - 1) & (gt_mot["id"] == t_id)]
    x = int((t["bb_left"] + t["bb_width"] // 2).values[0])
    y = int((t["bb_top"] + t["bb_height"] // 2).values[0])
    print(flow[y, x, :])

    t_current = gt_mot[(gt_mot["frame"] == i) & (gt_mot["id"] == t_id)]

    print(t, t_current)

    view_frame(frame2, gt_mot, i)
    view_flow(flow)

#%%
frame1 = get_frame("pNEUMA10/", 1)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

for i in range(2, 1000):
    frame2 = get_frame("pNEUMA10/", i)
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('frame', bgr)
    cv2.imshow('frame 2', frame2)
    if cv2.waitKey(1) == ord('q'):
        break
    frame1 = frame2

cv2.destroyAllWindows()

# %%
 
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 500,
    qualityLevel = 0.3,
    minDistance = 7,
    blockSize = 7 
)
 
# Parameters for lucas kanade optical flow
lk_params = dict( winSize = (15, 15),
    maxLevel = 2,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)
color = np.random.randint(0, 255, (500, 3))
t_mask = np.zeros((525, 1024), dtype="uint8")
t_mask[300:, :] = 1
for i in range(2, 1000):
    frame1 = get_frame("pNEUMA10/", i-1)
    frame2 = get_frame("pNEUMA10/", i)
    mask = np.zeros((525, 1024, 3), dtype='uint8')
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(prvs, mask = t_mask, **feature_params)
    p1, st, err = cv2.calcOpticalFlowPyrLK(prvs, next, p0, None, **lk_params)

    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        #frame = cv2.circle(frame2, (int(a), int(b)), 5, color[i].tolist(), -1)
        frame = frame2
        img = cv2.add(frame, mask)
    
    cv2.imshow('frame', img)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
# %%
