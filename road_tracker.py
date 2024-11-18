import cv2
import utils
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import filters as f
from typing import List

np.random.seed(0)

class RoadTracker():
    """Tracker which considers a vehicle in the road, following another car"""
    count = 1
    def __init__(self, det):
        self.id = RoadTracker.count
        self.curr_bbox = det
        self.last_bbox = det
        self.delta = np.array([0, 0])
        self.center = det[:2] + det[2:4] // 2
        self.last_center = self.center
        self.predict_count = 0
        self.history = []

        RoadTracker.count += 1

    def update(self, det):
        self.predict_count = 0

        self.curr_bbox = det
        self.last_center = self.center
        self.center = det[:2] + det[2:4] // 2
        self.delta = self.center - self.last_center
        self.history.append(det)

    def predict(self, nearest_trackers: List["RoadTracker"]):
        """Predicts the position of the vehicle based on the position
        of nearest vehicles."""
        self.predict_count += 1

    def get_boxes(self):
        """Returns the vehicle id and its current bounding box."""
        return (self.id, *self.curr_bbox)

def linear_assignment(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def get_detections(df: pd.DataFrame, frame: int) -> np.ndarray:
    return df[df["frame"] == frame].values[:, 2:6] #type: ignore

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

def find_nearest(p: np.ndarray, points: np.ndarray, n: int) -> np.ndarray:
    return ((p - points) ** 2).sum(axis=1).argsort()[1:n+1]

def find_leader(tracker: RoadTracker, near: List[RoadTracker]):
    dir = (tracker.center - tracker.last_center)
    if np.allclose(tracker.center, tracker.last_center):
        dir = np.array([1, 0])

    if dir[0] < 0:
        dir[0] = 1


    print(t.id, dir)

#gt_mot = u.load_mot("10_0900_0930_D10_RM_mot.txt")
#p_mot = u.load_mot("sort/output/pNEUMA10_8-tiny.txt")
#acc_mot = u.load_mot("sort-acc/output/pNEUMA10_8-tiny.txt")
det_mot = utils.load_mot_road("mots/yolo-tiny/cars-tiny-8.mot")

#frame1 = u.get_frame("pNEUMA10/", 1)
#u.view_frame(frame1, det_mot, 1)
# initialize all trackers
trackers = []
for det in get_detections(det_mot, 1):
    t = RoadTracker(det)
    trackers.append(t)


for i in range(2, 10):
    print(i)
    dets = get_detections(det_mot, i)
    ms, uds, uts = update(trackers, dets)

    current_trackers = []

    bboxes = get_boxes(trackers)
    centers = bboxes[:, 1:3] + bboxes[:, 3:] // 2
    for m in ms:
        t = trackers[m[1]]
        t.update(dets[m[0]])
        current_trackers.append(t)
        nearest = [trackers[i] for i in find_nearest(t.center, centers, 5)]
        find_leader(t, nearest)

    for u in uds:
        det = dets[u]
        t = RoadTracker(det)
        current_trackers.append(t)

    for u in uts:
        bboxes = get_boxes(trackers)
        centers = bboxes[:, 1:3] + bboxes[:, 3:] // 2
        print(f"nearest to {u}:", find_nearest(trackers[u].center, centers, 3))

    trackers = current_trackers

    frame2 = utils.get_frame("pNEUMA10/", i)
    utils.draw_all_rects(frame2, get_boxes(trackers))

    cv2.imshow("frame", frame2)
    cv2.waitKey(0)

cv2.destroyAllWindows()


