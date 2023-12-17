#%%
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


filename = "mots/city_above_mot.txt"

mot_data = {}

with open(filename) as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        frame, id, bb_left, bb_top, bb_width, bb_height, _, _, _, _ = map(
            float, row)
        frame = int(frame - 1)
        id = int(id)

        if id not in mot_data:
            mot_data[id] = {"frame": [], "bb_left": [], "bb_top": [],
            "w": [], "h": []}

        mot_data[id]["frame"].append(frame)
        mot_data[id]["bb_left"].append(bb_left)
        mot_data[id]["bb_top"].append(bb_top)
        mot_data[id]["w"].append(bb_width) 
        mot_data[id]["h"].append(bb_height)

# %%
plt.rcParams["figure.figsize"] = (10, 8)
fig, ax = plt.subplots()
ax.invert_yaxis()
for data in mot_data.values():
    ax.plot(data["bb_left"], data["bb_top"])
    for x, y, w, h in zip(data["bb_left"], data["bb_top"], data["w"], data["h"]):
        ax.add_patch(Rectangle((x,y - h/2), w, h, alpha=0.01))
plt.show()
# %%
plt.rcParams["figure.figsize"] = (10, 8)
fig, ax = plt.subplots()
ax.invert_yaxis()
for data in mot_data.values():
    ax.plot(data["bb_left"], data["bb_top"], 'o')
plt.show()

# %%
xs = [x for v in mot_data.values() for x in v['bb_left']]
ys = [y for v in mot_data.values() for y in v['bb_top']]
ws = [w for v in mot_data.values() for w in v['w']]
hs = [h for v in mot_data.values() for h in v['h']]

# %%
plt.rcParams["figure.figsize"] = (10, 8)
fig, ax = plt.subplots()
ax.invert_yaxis()
for data in mot_data.values():
    ax.plot(data["bb_left"], data["bb_top"])
    for x, y, w, h in zip(data["bb_left"], data["bb_top"], data["w"], data["h"]):
        ax.add_patch(Rectangle((x,y - h/2), w, h, alpha=0.1, edgecolor='c', facecolor='none'))
plt.show()

# %%
xs = [x for v in mot_data.values() for x in v['bb_left']]
ys = [y for v in mot_data.values() for y in v['bb_top']]
ws = [w for v in mot_data.values() for w in v['w']]
hs = [h for v in mot_data.values() for h in v['h']]

fig, ax = plt.subplots()
ax.invert_yaxis()
size = (int(np.max(xs)), int(np.max(ys)))

for data in mot_data.values():
    ax.plot(xs, ys, 'ro', markersize=0.1)
    
    
mhs = int(np.median(hs))
mws = int(np.median(ws))
data = {}
for y in range(0, size[1], mhs):
    print(y)
    for x in range(0, size[0], mws):
        ax.add_patch(
            Rectangle((x, y), mws, mhs, 
                        edgecolor='c', facecolor='none'))
        c = 0
        for i in range(len(xs)):
            c += xs[i] > x and xs[i] < x + mws \
                and ys[i] > y and ys[i] < y + mhs
        data[(x, y)] = c

plt.show()
# %%
fig, ax = plt.subplots()
ax.invert_yaxis()
ax.plot([0, 1], [0,1])
__k = np.array(list(data.values()))
__k = __k / __k.max()
for i, xy in enumerate(data):
    ax.add_patch(
            Rectangle((xy[0], xy[1]), mws, mhs, 
                        edgecolor='c', facecolor=(__k[i], 0, 0)))

# %%
from sklearn.cluster import DBSCAN
X = np.array([xs, ys]).T
db = DBSCAN(eps=10, metric='l1').fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
# %%
unique_labels = set(labels)
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

fig, ax = plt.subplots()
ax.invert_yaxis()
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = X[class_member_mask & core_samples_mask]

    ax.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )


ax.set_title(f"Estimated number of clusters: {n_clusters_}")
plt.show()

# %%
