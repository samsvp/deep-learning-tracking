#%%
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass

def load_mot(mot_file: str) -> pd.DataFrame:
    df = pd.read_csv(
        mot_file,
        names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"]
    )

    return df

def find_nearest(df: pd.DataFrame, id: int, frame: int):
    df_frame = df[df["frame"] == frame].copy()
    df_frame["xs"] = df_frame["bb_left"] + df_frame["bb_width"] / 2
    df_frame["ys"] = df_frame["bb_top"] + df_frame["bb_height"] / 2
    x = df_frame[df_frame["id"] == id]["xs"].values[0]
    y = df_frame[df_frame["id"] == id]["ys"].values[0]
    df_frame["dist"] = (df_frame["xs"] - x) ** 2 + (df_frame["ys"] - y) ** 2
    nearest = df_frame.sort_values(by='dist')
    return nearest[["frame", "id", "xs", "ys", "dist"]]

def find_nearest_2(df, id, frame, n):
    df_last_frame = df[df["frame"] == frame-1]
    if id not in df_last_frame["id"].values:
        return
    nearest_last = find_nearest(df, id, frame-1)
    nearest_current = find_nearest(df, id, frame)
    nearest = pd.merge(nearest_current, nearest_last, on='id', suffixes=('_c', '_l'))
    nearest = nearest.iloc[:n+1]
    print(nearest)


#%%
gt_mot = load_mot("10_0900_0930_D10_RM_mot.txt")

id = 16
n = 3
frame = 4
thresh = 10.0
find_nearest_2(gt_mot, id, frame, n)
# %%
