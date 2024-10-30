#%%
import numpy as np
import pandas as pd

def load_mot(mot_file: str) -> pd.DataFrame:
    df = pd.read_csv(
        mot_file,
        names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"]
    )

    # we are only interested in the bottom part of the file
    bottom_df = df[df["bb_top"] > 300].copy(deep=True)
    bottom_df["bb_top"] -= 300
    return bottom_df

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
        return [], []
    nearest_last = find_nearest(df, id, frame-1)
    nearest_current = find_nearest(df, id, frame)
    nearest = pd.merge(nearest_current, nearest_last, on='id', suffixes=('_c', '_l'))
    nearest = nearest.iloc[:n+1]
    v_l = nearest[["xs_l", "ys_l"]].values
    v_c = nearest[["xs_c", "ys_c"]].iloc[1:].values
    v = np.append(v_l, v_c, axis=0)
    t = nearest[["xs_c", "ys_c"]].values[0]
    return v, t


#%%
gt_mot = load_mot("10_0900_0930_D10_RM_mot.txt")

n = 7
frames = gt_mot["frame"].max()
frame_split = int(0.8 * frames)
print(f"Total train frames: {frame_split}")
vs = np.array([])
ts = np.array([])
for frame in range(2, frame_split):
    gt_frame = gt_mot[gt_mot["frame"] == frame]
    for id in gt_frame["id"].values:
        v, t = find_nearest_2(gt_mot, id, frame, n)
        if len(v) == 0:
            continue

        if vs.shape[0] != 0:
            vs = np.dstack((vs, v))
            ts = np.dstack((ts, t))
        else:
            vs = v 
            ts = t

    if frame % 10 == 0:
        print(f"Finished frame {frame}")

#%%
print(f"Test frames: {frame_split} to {frames}")
vs_test = np.array([])
ts_test = np.array([])
for frame in range(frame_split, frames):
    gt_frame = gt_mot[gt_mot["frame"] == frame]
    for id in gt_frame["id"].values:
        v, t = find_nearest_2(gt_mot, id, frame, n)
        if len(v) == 0:
            continue

        if vs_test.shape[0] != 0:
            vs_test = np.dstack((vs_test, v))
            ts_test = np.dstack((ts_test, t))
        else:
            vs_test = v 
            ts_test = t

    if frame % 10 == 0:
        print(f"Finished frame {frame}")

#%%
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
X = vs.reshape((n * 2 + 1) * 2, -1).T  
y = ts.reshape(-1, 2)
model = MultiOutputRegressor(RandomForestRegressor())
reg = model.fit(X, y)
# %%
X_test = vs_test.reshape((n * 2 + 1) * 2, -1).T 
y_test = ts_test.reshape(-1, 2)
reg.score(X_test, y_test)

#%%
from sklearn.neural_network import MLPRegressor
regr = MLPRegressor(random_state=1, max_iter=1000).fit(X, y)
regr.score(X, y)
# %%
regr.score(X_test, y_test)

# %%
from sklearn import svm
regr = MultiOutputRegressor(svm.SVR())
regr.fit(X, y)
print(regr.score(X, y))
print(regr.score(X_test, y_test))
# %%
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler
model = MultiOutputRegressor(SGDRegressor(max_iter=10_000, tol=1e-3))
reg = make_pipeline(StandardScaler(), model)
reg.fit(X, y)
print(reg.score(X, y))
print(reg.score(X_test, y_test))
# %%
