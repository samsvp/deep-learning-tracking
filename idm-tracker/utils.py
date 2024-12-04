import pandas as pd
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

def convert_det_to_z(det: np.ndarray) -> np.ndarray:
    """
    Takes a bounding box in the form [x1,y1,w,h] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
    """
    w = det[2]
    h = det[3]
    x = det[0] + w/2.
    y = det[1] + h/2.
    s = w * h    #scale is just area
    r = w / float(h)
    return np.array([x, y, s, r])

def convert_x_to_det(x):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,w,h] where x1,y1 is the top left and w, h are the width and height
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    return np.array([x[0] - w / 2., x[1] - h / 2., w, h])

def create_UKF(x0):
    sigmas = MerweScaledSigmaPoints(8, alpha=.1, beta=2., kappa=1.)
    def fx(x, _):
        return x

    def hx(x):
        return x[:4]

    ukf = UKF(dim_x=8, dim_z=4, fx=fx,
              hx=hx, dt=1, points=sigmas)
    ukf.x = x0
    ukf.R = np.diag([100.0, 100.0, 100.0, 100.0]) / 2
    ukf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    ukf.P *= 10.
    ukf.Q[:2, :2] *= 0.1
    ukf.Q[-1,-1] *= 10
    ukf.Q[4:,4:] *= 10
    return ukf

def line_intersects_rect(p1, p2, r):
    return line_intersects(p1, p2, np.array([r[0], r[1]]), np.array([r[2], r[1]])) or\
           line_intersects(p1, p2, np.array([r[2], r[1]]), np.array([r[2], r[3]])) or\
           line_intersects(p1, p2, np.array([r[2], r[3]]), np.array([r[0], r[3]])) or\
           line_intersects(p1, p2, np.array([r[0], r[3]]), np.array([r[0], r[1]]))

def line_intersects(l1p1, l1p2, l2p1, l2p2):
    q = (l1p1[1] - l2p1[1]) * (l2p2[0] - l2p1[0]) - (l1p1[0] - l2p1[0]) * (l2p2[1] - l2p1[1]);
    d = (l1p2[0] - l1p1[0]) * (l2p2[1] - l2p1[1]) - (l1p2[1] - l1p1[1]) * (l2p2[0] - l2p1[0]);

    if d == 0:
        return False

    r = q / d

    q = (l1p1[1] - l2p1[1]) * (l1p2[0] - l1p1[0]) - (l1p1[0] - l2p1[0]) * (l1p2[1] - l1p1[1])
    s = q / d

    return not( (r < 0) or (r > 1) or (s < 0) or (s > 1) )

def load_mot_road(mot_file: str) -> pd.DataFrame:
    df = pd.read_csv(
        mot_file,
        names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"]
    )

    # we are only interested in the bottom part of the file
    bottom_df = df[df["bb_top"] > 350].copy(deep=True)

    return bottom_df #type: ignore
