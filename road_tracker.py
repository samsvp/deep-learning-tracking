import cv2
import utils
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from typing import List

np.random.seed(0)
# good targets with continuous detection: 23, 24, 27
# good targets with break: 1, 3, 24, 33

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

class RoadTracker():
    #insert unscented kalman filter
    """Tracker which considers a vehicle in the road, following another car"""
    count = 1
    def __init__(self, det, is_mock=False):
        self.curr_bbox = det
        self.last_bbox = det
        self.delta = np.array([0, 0])
        self.center = det[:2] + det[2:4] // 2
        self.last_center = self.center
        self.predict_count = 0
        self.det_history = []
        self.pred_history = []
        self.det_pred_map = []
        self.leaders = {}
        self.followers = {}
        z = convert_det_to_z(det)
        x0 = np.array([z[0], z[1], z[2], z[3], 0, 0, 0, 0])
        self.ukf = create_UKF(x0)

        if not is_mock:
            self.id = RoadTracker.count
            RoadTracker.count += 1
        else:
            self.id = -1

    def update(self, det):
        self.predict_count = 0

        z = convert_det_to_z(det)
        self.ukf.update(z)

        p_det = convert_x_to_det(self.ukf.x[:4])
        self.curr_bbox = p_det
        self.center = p_det[:2] + p_det[2:4] // 2
        self.delta = self.center - self.last_center
        self.det_history.append(p_det)
        self.det_pred_map.append((
            len(self.det_history) - 1,
            len(self.pred_history) - 1,
        ))

    def predict(self, nearest_trackers: List["RoadTracker"]) -> np.ndarray:
        """Predicts the position of the vehicle based on the position
        of nearest vehicles."""
        self.predict_count += 1
        def fx(x, _):
            self.center = x[:2]
            self.leaders, follower = find_leader_follower(self, nearest_trackers)
            acc = idm(self)
            v = self.center - self.last_center
            center = self.center + v + acc / 2
            new_v = v + acc
            if new_v[0] < 0:
                new_v[0] = 0
                center[0] = self.center[0]
            new_x = np.array([center[0], center[1], x[2], x[3], new_v[0], new_v[1], acc[0], acc[1]])
            return new_x

        try:
            self.ukf.predict(fx=fx)
        except Exception as e:
            print(f"Error on {self.id}: {e}")
            raise e
        x = self.ukf.x
        self.last_center = self.center
        self.center = x[:2]
        p_det = convert_x_to_det(x[:4])
        self.curr_bbox = p_det
        self.pred_history.append(p_det)
        return x

    def get_boxes(self):
        """Returns the vehicle id and its current bounding box."""
        id = self.id if self.predict_count == 0 else -self.id
        return (id, *self.curr_bbox)

    def get_pred_boxes(self):
        if len(self.pred_history) > 0:
            return (-self.id, *self.pred_history[-1])
        else:
            return []

    def get_leader(self):
        max_score = 0
        c_leader = None
        for leader, score in self.leaders.items():
            if score < max_score: continue

            c_leader = leader
            max_score = score
        return c_leader

class Trackers():
    def __init__(self, dets: np.ndarray) -> None:
        self.current = [RoadTracker(det) for det in dets]
        self.history = [[c.get_boxes() for c in self.current]]

    def predict(self):
        bboxes = get_boxes(self.current)
        centers = bboxes[:, 1:3] + bboxes[:, 3:] // 2
        for t in self.current:
            nearest = [self.current[i] for i in find_nearest(t.center, centers, 5)]
            t.predict(nearest)

    def update(self, dets: np.ndarray, limit_x = 1000, max_age = 5):
        ms, uds, uts = update(self.current, dets)

        current_trackers = []

        for m in ms:
            t = self.current[m[1]]
            t.update(dets[m[0]])
            current_trackers.append(t)

        for u in uds:
            det = dets[u]
            t = RoadTracker(det)
            current_trackers.append(t)

        bboxes = get_boxes(self.current)
        centers = bboxes[:, 1:3] + bboxes[:, 3:] // 2
        current_trackers.sort(key=lambda t: t.id)
        for t in current_trackers:
            nearest = [self.current[i] for i in find_nearest(t.center, centers, 5)]
            t.leaders, t.followers = find_leader_follower(t, nearest)

        for ut in uts:
            t = self.current[ut]
            x = t.ukf.x
            # offscreen
            if x[0] > limit_x or x[0] < 0:
                continue

            # too few frames, ignore
            if len(t.det_history) < 2:
                continue

            if t.predict_count < max_age:
                current_trackers.append(t)

        self.current = current_trackers
        self.history.append([c.get_boxes() for c in self.current])

    def to_mot(self):
        lines = []
        for i, trackers in enumerate(self.history):
            for tracker in trackers:
                if tracker[0] < 0:
                    continue
                line = f"{i+1},{','.join((str(t) for t in tracker))},-1,-1,-1,-1"
                lines.append(line)
        return "\n".join(lines)

def linear_assignment(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def get_detections(df: pd.DataFrame, frame: int) -> np.ndarray:
    return df[df["frame"] == frame].values[:, 2:7] #type: ignore

def iou_batch(bb_test, bb_gt):
    """From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]"""
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return(o)

def update(trackers: List[RoadTracker], dets: np.ndarray, iou_threshold=0.05):
    boxes = np.array([t.curr_bbox for t in trackers])
    dets = dets.copy()
    boxes[:, 2:] += boxes[:, :2]
    dets[:, 2:] += dets[:, :2]
    iou_matrix = iou_batch(dets, boxes)
    matched = linear_assignment(-iou_matrix)
    unmatched_detections = []
    for d, _ in enumerate(dets):
        if(d not in matched[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, _ in enumerate(trackers):
        if(t not in matched[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched:
        if iou_matrix[m[0], m[1]]<iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if len(matches) == 0:
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)
    return matches, unmatched_detections, unmatched_trackers

def get_boxes(trackers: List[RoadTracker]) -> np.ndarray:
    return np.array([t.get_boxes() for t in trackers])

def get_pred_boxes(trackers: List[RoadTracker]) -> np.ndarray:
    return np.array([
        t.get_pred_boxes() for t in trackers
        if len(t.pred_history) > 0
    ])

def find_nearest(p: np.ndarray, points: np.ndarray, n: int) -> np.ndarray:
    return ((p - points) ** 2).sum(axis=1).argsort()[1:n+1]


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

def find_dir(tracker: RoadTracker, near: List[RoadTracker], dir: np.ndarray):
    dir /= np.sum(dir ** 2) ** 0.5
    leaders = {}
    alpha = 100
    noises = [np.array([0.05, 0.05]), np.array([0.0, 0]), np.array([-0.05, -0.05])]
    for noise in noises:
        p1 = tracker.center
        p2 = p1 + alpha * (dir + noise)
        for n in near:
            rect = n.curr_bbox.copy()
            rect[2:] += rect[:2]
            if not line_intersects_rect(p1, p2, rect):
                continue

            score = leaders.get(n, 0) + 1
            leaders[n] = score

    return leaders

def find_leader_follower(tracker: RoadTracker, near: List[RoadTracker]):
    dir = (tracker.center - tracker.last_center)
    if np.allclose(tracker.center, tracker.last_center):
        dir = np.array([1.0, 0])

    if dir[0] < 0:
        dir[0] = 1.0

    return find_dir(tracker, near, dir), find_dir(tracker, near, -dir)


def idm(vehicle: RoadTracker, v0=15, T=1.0, s0=12, a=1.0, b=0.4):
    leader = vehicle.get_leader()
    if leader is None:
        leader = RoadTracker(np.array([1000, 1000, 20, 20]), is_mock=True)

    v = vehicle.center[:2] - vehicle.last_center[:2]
    vl = leader.center[:2] - leader.last_center[:2]
    dv = v - vl
    ds = ((leader.center[:2] - vehicle.center[:2]) ** 2).sum()
    if abs(ds) < 0.01:
        print(vehicle.id, leader.id)
    s_star = s0 + np.maximum(0, (v * T + v * dv / (2 * np.sqrt(a * b))))
    acc = a * (1 - (v / v0) ** 4 - (s_star / ds) ** 2)
    return np.maximum(np.minimum(acc, a), -a)


def suppress_detections(dets: np.ndarray, thresh=0.7) -> np.ndarray:
    i = 0
    bb_dets = dets.copy()
    bb_dets[:, 2:4] += bb_dets[:, :2]
    while i < len(dets) - 1:
        det = bb_dets[i]
        iou_matrix = iou_batch(np.array([det]), bb_dets[i+1:])
        double_indexes = np.argwhere(iou_matrix > thresh).reshape(-1) + i + 1
        if len(double_indexes) == 0:
            i += 1
            continue

        a_max = np.argmax(bb_dets[double_indexes][:, -1])
        # bigger confidence on a_max
        remove_idx = \
            [d for d in double_indexes if d != a_max] + [i] \
            if bb_dets[a_max][-1] > det[-1] \
            else double_indexes

        # delete the detections with smaller confidence
        dets = np.delete(dets, remove_idx, axis=0)
        bb_dets = np.delete(bb_dets, remove_idx, axis=0)
        i += 1

    return dets[:, :4]

#gt_mot = u.load_mot("10_0900_0930_D10_RM_mot.txt")
#p_mot = u.load_mot("sort/output/pNEUMA10_8-tiny.txt")
#acc_mot = u.load_mot("sort-acc/output/pNEUMA10_8-tiny.txt")
det_mot = utils.load_mot_road("mots/yolo-tiny/cars-tiny-8.mot")

#frame1 = u.get_frame("pNEUMA10/", 1)
#u.view_frame(frame1, det_mot, 1)
# initialize all trackers
dets = get_detections(det_mot, 1)
dets = suppress_detections(dets, thresh=0.3)
trackers = Trackers(dets)

for i in range(2, 1908):
    print(i)
    dets = get_detections(det_mot, i)
    dets = suppress_detections(dets, thresh=0.3)

    trackers.predict()
    trackers.update(dets, max_age=5)

    """
    frame = utils.get_frame("pNEUMA10/", i)
    frame_det = frame.copy()
    utils.draw_all_rects(frame, get_boxes(trackers.current))
    utils.draw_all_rects(frame_det, get_pred_boxes(trackers.current))
    #utils.draw_all_rects(frame_det, [(1, *det) for det in dets]) #type: ignore

    cv2.imshow("frame det", frame_det)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

    cv2.destroyAllWindows()
    """

mot = trackers.to_mot()
with open("road.mot", 'w') as fp:
    fp.write(mot)

print("done")
