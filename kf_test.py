#%%
from __future__ import print_function

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter, EnsembleKalmanFilter
from filterpy.stats import mahalanobis
import utils as u

np.random.seed(0)

def convert_det_bbox(det):
    w = det[2]
    h = det[3]
    x1 = det[0]
    y1 = det[1]
    x2 = x1 + w 
    y2 = y1 + h 
    return np.array([x1, y1, x2, y2])

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    #scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

def calc_optical_flow(frame1, frame2):
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prvs, 
        next, 
        None, # type: ignore
        0.5, 
        3, 
        15, 
        3, 
        5, 
        1.2,
        0,
    )
    return flow 

def get_flow_bbox(flow, bboxes):
    """
    Gets the flow inside the bounding box [x1, y1, x2, y2]
    """
    bboxes = bboxes.astype(int)
    return [np.mean(
        flow[bbox[1]:bbox[3], bbox[0]:bbox[2], :],
        axis=(0, 1)
    ) for bbox in bboxes]


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    self.kf.F = np.array([
        [1,0,0,0,1,0,0],
        [0,1,0,0,0,1,0],
        [0,0,1,0,0,0,1],
        [0,0,0,1,0,0,0],  
        [0,0,0,0,1,0,0],
        [0,0,0,0,0,1,0],
        [0,0,0,0,0,0,1]
    ])
    self.kf.H = np.array([
        [1,0,0,0,0,0,0],
        [0,1,0,0,0,0,0],
        [0,0,1,0,0,0,0],
        [0,0,0,1,0,0,0]
    ])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
        self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
        self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)

class KalmanBoxTrackerAcc(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self,bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=9, dim_z=4) 
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0.5, 0  ],
            [0, 1, 0, 0, 0, 1, 0, 0,   0.5],
            [0, 0, 1, 0, 0, 0, 1, 0,   0  ],
            [0, 0, 0, 1, 0, 0, 0, 0,   0  ],  
            [0, 0, 0, 0, 1, 0, 0, 1,   0  ],
            [0, 0, 0, 0, 0, 1, 0, 0,   1  ],
            [0, 0, 0, 0, 0, 0, 1, 0,   0  ],
            [0, 0, 0, 0, 0, 0, 0, 1,   0  ],
            [0, 0, 0, 0, 0, 0, 0, 0,   1  ],
        ])
        self.kf.H = np.array([
            [1,0,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0,0],
            [0,0,0,1,0,0,0,0,0]
        ])

        self.pos_R = np.array([
            [ 1.,  0.,  0.,  0.],
            [ 0.,  1.,  0.,  0.],
            [ 0.,  0., 10.,  0.],
            [ 0.,  0.,  0., 10.]
        ])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self,bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))


    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


class KalmanBoxTrackerAdaptative(object):
    count = 0

    def __init__(self,bbox):
        self.kf_cv = KalmanBoxTracker(bbox)
        self.kf_acc = KalmanBoxTrackerAcc(bbox)
        self.current_model = self.kf_cv

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def predict(self):
        self.kf_cv.predict()
        self.kf_acc.predict()

        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.current_model.kf.x))
        return self.history[-1]
    
    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

        kf_acc_state = self.kf_acc.get_state()
        kf_cv_state = self.kf_cv.get_state()

        dcv = np.sum((bbox[:4].reshape(1, 4) - kf_cv_state) ** 2)
        dacc = np.sum((bbox[:4].reshape(1, 4) - kf_acc_state) ** 2)

        self.kf_acc.update(bbox)
        self.kf_cv.update(bbox)

        if dcv < dacc:
            self.current_model = self.kf_cv
        else:
            self.current_model = self.kf_acc

    def get_state(self):
        return self.current_model.get_state()


class KalmanBoxTrackerWithVelocity(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self,bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=9, dim_z=4) 
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0.5, 0  ],
            [0, 1, 0, 0, 0, 1, 0, 0,   0.5],
            [0, 0, 1, 0, 0, 0, 1, 0,   0  ],
            [0, 0, 0, 1, 0, 0, 0, 0,   0  ],  
            [0, 0, 0, 0, 1, 0, 0, 1,   0  ],
            [0, 0, 0, 0, 0, 1, 0, 0,   1  ],
            [0, 0, 0, 0, 0, 0, 1, 0,   0  ],
            [0, 0, 0, 0, 0, 0, 0, 1,   0  ],
            [0, 0, 0, 0, 0, 0, 0, 0,   1  ],
        ])

        self.pos_H = np.array([
            [1,0,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0,0],
            [0,0,0,1,0,0,0,0,0]
        ])
        self.vel_H = np.array([
            [0,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,1,0,0,0],
        ])
        self.kf.H = self.pos_H

        self.pos_R = np.array([
            [ 1.,  0.,  0.,  0.],
            [ 0.,  1.,  0.,  0.],
            [ 0.,  0., 10.,  0.],
            [ 0.,  0.,  0., 10.]
        ])

        self.vel_R = np.array([
            [ 10.,  0.],
            [ 0.,  10.],
        ])

        self.kf.R = self.pos_H
        self.kf.P[4:,4:] *= 10. #give high uncertainty to the unobservable initial velocities
        self.kf.P[7:,7:] *= 100.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.Q[7:,7:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox=None, vel=None):
        """
        Updates the state vector with observed bbox.
        """
        if vel is not None:
            self._update_vel(vel)
        if bbox is not None:
            self._update_pos(bbox)

    def _update_pos(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.H = self.pos_H
        self.kf.R = self.pos_R
        self.kf.dim_z = 4
        self.kf.update(convert_bbox_to_z(bbox))

    def _update_vel(self, vel):
        self.kf.H = self.vel_H
        self.kf.R = self.vel_R
        self.kf.dim_z = 2
        self.kf.update(vel)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

def get_bbox(i, dets):
   return convert_det_bbox(dets.values[i][2:6])

class ParticleFilter(object):
    def __init__(self,bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0.5, 0  ],
            [0, 1, 0, 0, 0, 1, 0, 0,   0.5],
            [0, 0, 1, 0, 0, 0, 1, 0,   0  ],
            [0, 0, 0, 1, 0, 0, 0, 0,   0  ],  
            [0, 0, 0, 0, 1, 0, 0, 1,   0  ],
            [0, 0, 0, 0, 0, 1, 0, 0,   1  ],
            [0, 0, 0, 0, 0, 0, 1, 0,   0  ],
            [0, 0, 0, 0, 0, 0, 0, 1,   0  ],
            [0, 0, 0, 0, 0, 0, 0, 0,   1  ],
        ])

        self.H = np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]
        ])

        P = np.eye(9) * 1_000
        x = np.zeros(9)
        x[:4] = convert_bbox_to_z(bbox).reshape(-1)

        self.pf = EnsembleKalmanFilter(
            x=x, 
            P=P, 
            dim_z=4,
            dt=1, 
            N=10_000,
            fx=self.fx,
            hx=self.fh,
        ) 
        self.R = np.array([
            [ 1.,  0.,  0.,  0.],
            [ 0.,  1.,  0.,  0.],
            [ 0.,  0., 10.,  0.],
            [ 0.,  0.,  0., 10.]
        ])

        self.pf.R = self.R
        self.pf.Q[-1,-1] *= 0.01
        self.pf.Q[4:,4:] *= 0.01
        self.pf.Q[7:,7:] *= 0.01


        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def fx(self, x, _):
        return np.dot(self.F, x)

    def fh(self, x):
        return x[:4]

    def update(self,bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.pf.update(convert_bbox_to_z(bbox).reshape(-1))


    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.pf.x[6]+self.pf.x[2])<=0):
            self.pf.x[6] *= 0.0
        self.pf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.pf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.pf.x)


def mahalanobis_batch(x, mean, V):
    VI = np.linalg.pinv(V)
    d = x - mean
    return np.sqrt(np.einsum('ij,ij->i', np.einsum('ij,kj->ik', d,VI), d))


#%%
gt_mot = u.load_mot("10_0900_0930_D10_RM_mot.txt")
p_mot = u.load_mot("sort/output/pNEUMA10_8-tiny.txt")
det_mot = u.load_mot("mots/yolo-tiny/cars-tiny-8.mot")
acc_mot = u.load_mot("sort-acc/output/pNEUMA10_8-tiny.txt")

frame1 = u.get_frame("pNEUMA10/", 1)
frame2 = u.get_frame("pNEUMA10/", 2)
u.view_frame(frame1, p_mot, 1)
u.view_frame(frame1, gt_mot, 1)

frame1 = u.get_frame("pNEUMA10/", 25)
frame2 = u.get_frame("pNEUMA10/", 26)
flow = calc_optical_flow(frame1, frame2)
u.view_flow_bbox(flow, gt_mot, 25)

# %%
#target_p = 33
#target_gt = 39
target_p = 34
target_gt = 189

track_gt = gt_mot[gt_mot["id"] == target_gt]
plt.plot(track_gt["bb_left"], 'o', label="gt")

track_p = p_mot[p_mot["id"] == target_p]
plt.plot(track_p["bb_left"], 'o', label="pred")

plt.legend()
# %%
initial_bbox = get_bbox(0, track_gt)
tracker = KalmanBoxTracker(initial_bbox)
initial_bbox_p = get_bbox(0, track_p)
tracker_acc = KalmanBoxTrackerAcc(initial_bbox_p)
tracker_adp = KalmanBoxTrackerAdaptative(initial_bbox_p)
tracker_vel = KalmanBoxTrackerWithVelocity(initial_bbox_p)
tracker_pf = ParticleFilter(initial_bbox_p)

positions = [tracker.get_state()[0]]
positions_preds = [tracker.get_state()[0]]
positions_acc = [tracker_acc.get_state()[0]]
positions_acc_preds = [tracker_acc.get_state()[0]]
positions_vel = [tracker_vel.get_state()[0]]
positions_vel_preds = [tracker_vel.get_state()[0]]
positions_adp = [tracker_adp.get_state()[0]]
positions_adp_preds = [tracker_adp.get_state()[0]]
positions_pf = [tracker_adp.get_state()[0]]
positions_pf_preds = [tracker_adp.get_state()[0]]
for i in range(1, len(track_p)):
    # calc vel
    frame1 = u.get_frame("pNEUMA10/", i)
    frame2 = u.get_frame("pNEUMA10/", i+1)
    flow = calc_optical_flow(frame1, frame2)
    box_p = get_bbox(i, track_p)
    vel_p = get_flow_bbox(flow, np.array([box_p]))

    # calc pos
    pred_pos = tracker_acc.predict()[0]
    positions_acc_preds.append(pred_pos)
    tracker_acc.update(box_p)
    pos = tracker_acc.get_state()[0]
    positions_acc.append(pos)

    pred_pos = tracker_adp.predict()[0]
    positions_adp_preds.append(pred_pos)
    tracker_adp.update(box_p)
    pos_adp = tracker_adp.get_state()[0]
    positions_adp.append(pos_adp)

    pred_pf = tracker_pf.predict()[0]
    positions_pf_preds.append(pred_pf)
    tracker_pf.update(box_p)
    pos_pf = tracker_pf.get_state()[0]
    positions_pf.append(pos_pf)

    pred_vel = tracker_vel.predict()[0]
    positions_vel_preds.append(pred_vel)
    tracker_vel.update(box_p, vel_p)
    pos = tracker_vel.get_state()[0]
    positions_vel.append(pos)

    pos_pred = tracker.predict()[0]
    tracker.update(get_bbox(i, track_gt))
    pos = tracker.get_state()[0]
    positions.append(pos)
    positions_preds.append(pos_pred)


for i in range(len(track_p), len(track_p) + 10):
    tracker.predict()
    pos = tracker.get_state()[0]
    positions.append(pos)

    tracker_acc.predict()
    pos_p = tracker_acc.get_state()[0]
    positions_acc.append(pos_p)
    
    tracker_adp.predict()
    pos_adp = tracker_adp.get_state()[0]
    positions_adp.append(pos_adp)

    tracker_pf.predict()
    pos_pf = tracker_pf.get_state()[0]
    positions_pf.append(pos_pf)

    # calc vel
    tracker_vel.predict()
    pos_vel = tracker_vel.get_state()[0]
    frame1 = u.get_frame("pNEUMA10/", i)
    frame2 = u.get_frame("pNEUMA10/", i+1)
    flow = calc_optical_flow(frame1, frame2)
    vel_p = get_flow_bbox(flow, np.array([pos_vel]))
    tracker_vel.update(vel=vel_p)
    pos_vel = tracker_vel.get_state()[0]

    positions_vel.append(pos_vel)

track_gt = gt_mot[gt_mot["id"] == target_gt]
track_p = p_mot[p_mot["id"] == target_p]

N = 30
plt.plot(track_gt["bb_left"].values[:N], 'o', label="gt") #type: ignore
plt.plot(track_p["bb_left"].values, 'o', label="pred") #type: ignore
plt.plot([p[0] for p in positions_adp], 'o', label="kf adp")
plt.plot([p[0] for p in positions_adp_preds], 'o', label="kf adp preds")
plt.legend()
plt.show()

plt.plot(track_gt["bb_left"].values[:N], 'o', label="gt") #type: ignore
plt.plot(track_p["bb_left"].values, 'o', label="pred") #type: ignore
plt.plot([p[0] for p in positions_vel], 'o', label="kf vel")
plt.plot([p[0] for p in positions_vel_preds], 'o', label="kf vel preds")
plt.legend()
plt.show()

plt.plot(track_gt["bb_left"].values[:N], 'o', label="gt") #type: ignore
plt.plot(track_p["bb_left"].values, 'o', label="pred") #type: ignore
plt.plot([p[0] for p in positions_acc], 'o', label="kf p acc")
plt.plot([p[0] for p in positions_acc_preds], 'o', label="kf preds acc")
plt.legend()
plt.show()

plt.plot(track_gt["bb_left"].values[:N], 'o', label="gt") #type: ignore
plt.plot(track_p["bb_left"].values, 'o', label="pred") #type: ignore
plt.plot([p[0] for p in positions_pf], 'o', label="kf p pf")
plt.plot([p[0] for p in positions_pf_preds], 'o', label="kf preds pf")
plt.legend()
plt.show()

# %%
track_gt = gt_mot[gt_mot["id"] == target_gt]
plt.plot(track_gt["bb_top"].values, 'o', label="gt") #type: ignore

track_p = p_mot[p_mot["id"] == target_p]
plt.plot(track_p["bb_top"].values, 'o', label="pred") #type: ignore

plt.plot([p[1] for p in positions], 'o', label="kf gt")
plt.plot([p[1] for p in positions_acc], 'o', label="kf p")
plt.plot([p[1] for p in positions_vel], 'o', label="kf vel")

plt.legend()
# %%
u.play(acc_mot)
# %%
for i, (p_d, p, p_vel) in enumerate(
        zip(positions, positions_acc, positions_vel)
    ):
    id_ = 33 if i < len(track_p) else 34
    p_d = p_d.astype(int)
    p = p.astype(int)
    p_vel = p_vel.astype(int)
    frame = u.get_frame("pNEUMA10/", i+1)
    w = p[2] - p[0]
    h = p[3] - p[1]
    bb_p = u.BB(p[0] + w // 2, p[1] + h // 2, w, h)
    u.draw_rect(frame, bb_p, id_)
    
    w = p_d[2] - p_d[0]
    h = p_d[3] - p_d[1]
    bb_d = u.BB(p_d[0] + w // 2, p_d[1] + h // 2, w, h)
    u.draw_rect(frame, bb_d, id_ + 10)

    w = p_vel[2] - p_vel[0]
    h = p_vel[3] - p_vel[1]
    bb_vel = u.BB(p_vel[0] + w // 2, p_vel[1] + h // 2, w, h)
    u.draw_rect(frame, bb_vel, id_ + 20)

    cv2.imshow('frame', frame)
    if cv2.waitKey(0) == ord('q'):
        break

cv2.destroyAllWindows()
# %%
