import cv2
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter

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

def convert_det_bbox(det):
    w = det[2]
    h = det[3]
    x1 = det[0]
    y1 = det[1]
    x2 = x1 + w
    y2 = y1 + h
    return np.array([x1, y1, x2, y2])

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
            [ 20.,  0.],
            [ 0.,  20.],
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
