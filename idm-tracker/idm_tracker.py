import utils
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List

class VehicleTracker():
    """Tracker which considers a vehicle in the road, following another car"""
    count = 1
    def __init__(self, det, is_mock=False):
        self.curr_bbox = det[:4]
        self.last_bbox = det[:4]
        self.delta = np.array([0, 0])
        self.center = det[:2] + det[2:4] // 2
        self.last_center = self.center
        self.predict_count = 0
        self.det_history = []
        self.pred_history = []
        self.det_pred_map = []
        self.leaders = {}
        self.followers = {}
        z = utils.convert_det_to_z(det)
        x0 = np.array([z[0], z[1], z[2], z[3], 0, 0, 0, 0])
        self.ukf = utils.create_UKF(x0)

        if not is_mock:
            self.id = VehicleTracker.count
            VehicleTracker.count += 1
        else:
            self.id = -1

    def update(self, det):
        self.predict_count = 0

        z = utils.convert_det_to_z(det)
        self.ukf.update(z)

        p_det = utils.convert_x_to_det(self.ukf.x[:4])
        self.curr_bbox = p_det
        self.center = p_det[:2] + p_det[2:4] // 2
        self.delta = self.center - self.last_center
        self.det_history.append(p_det)
        self.det_pred_map.append((
            len(self.det_history) - 1,
            len(self.pred_history) - 1,
        ))

    def predict(self, nearest_trackers: List["VehicleTracker"]) -> np.ndarray:
        """Predicts the position of the vehicle based on the position
        of nearest vehicles."""
        self.predict_count += 1
        def fx(x, _):
            self.center = x[:2]
            self.leaders, _ = self.find_leader_follower(nearest_trackers)
            acc = self.idm()
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
        p_det = utils.convert_x_to_det(x[:4])
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

    def find_dir(self, near: List["VehicleTracker"], dir: np.ndarray):
        dir /= np.sum(dir ** 2) ** 0.5
        leaders = {}
        alpha = 100
        noises = [np.array([0.05, 0.05]), np.array([0.0, 0]), np.array([-0.05, -0.05])]
        for noise in noises:
            p1 = self.center
            p2 = p1 + alpha * (dir + noise)
            for n in near:
                rect = n.curr_bbox.copy()
                rect[2:] += rect[:2]
                if not utils.line_intersects_rect(p1, p2, rect):
                    continue

                score = leaders.get(n, 0) + 1
                leaders[n] = score

        return leaders

    def find_leader_follower(self, near: List["VehicleTracker"]):
        dir = (self.center - self.last_center)
        if np.allclose(self.center, self.last_center):
            dir = np.array([1.0, 0])

        if dir[0] < 0:
            dir[0] = 1.0

        return self.find_dir(near, dir), self.find_dir(near, -dir)


    def idm(self, v0=15, T=1.0, s0=12, a=1.0, b=0.4):
        leader = self.get_leader()
        if leader is None:
            leader = VehicleTracker(np.array([1000, 1000, 20, 20]), is_mock=True)

        v = self.center[:2] - self.last_center[:2]
        vl = leader.center[:2] - leader.last_center[:2]
        dv = v - vl
        ds = ((leader.center[:2] - self.center[:2]) ** 2).sum()
        if abs(ds) < 0.01:
            print(self.id, leader.id)
        s_star = s0 + np.maximum(0, (v * T + v * dv / (2 * np.sqrt(a * b))))
        acc = a * (1 - (v / v0) ** 4 - (s_star / ds) ** 2)
        return np.maximum(np.minimum(acc, a), -a)


class Trackers():
    def __init__(self, dets: np.ndarray) -> None:
        dets = Trackers.suppress_detections(dets)
        self.current_det = [VehicleTracker(det) for det in dets]
        self.current = [VehicleTracker(det) for det in dets]
        self.history = [[c.get_boxes() for c in self.current]]

    def predict(self):
        def find_nearest(p: np.ndarray, points: np.ndarray, n: int) -> np.ndarray:
            return ((p - points) ** 2).sum(axis=1).argsort()[1:n+1]

        bboxes = np.array([t.get_boxes() for t in self.current_det])
        centers = bboxes[:, 1:3] + bboxes[:, 3:] // 2
        for t in self.current_det:
            nearest = [self.current_det[i] for i in find_nearest(t.center, centers, 5)]
            t.predict(nearest)

    @staticmethod
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

    @staticmethod
    def suppress_detections(dets: np.ndarray, thresh=0.7) -> np.ndarray:
        i = 0
        bb_dets = dets.copy()
        bb_dets[:, 2:4] += bb_dets[:, :2]
        while i < len(dets) - 1:
            det = bb_dets[i]
            iou_matrix = Trackers.iou_batch(np.array([det]), bb_dets[i+1:])
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

    def match_dets(self, dets: np.ndarray, iou_threshold=0.05):
        dets = Trackers.suppress_detections(dets)
        trackers = self.current_det
        boxes = np.array([t.curr_bbox for t in trackers])
        dets = dets.copy()
        boxes[:, 2:] += boxes[:, :2]
        dets[:, 2:] += dets[:, :2]
        iou_matrix = Trackers.iou_batch(dets, boxes)

        x, y = linear_sum_assignment(-iou_matrix)
        matched = np.array(list(zip(x, y)))

        unmatched_detections = []
        for d, _ in enumerate(dets):
            if d not in matched[:,0]:
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t, _ in enumerate(trackers):
            if t not in matched[:,1]:
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

        i = len(unmatched_detections)
        for u in reversed(unmatched_detections):
            if sum(iou_matrix[u] > iou_threshold) >= 1:
                unmatched_detections.pop(i-1)
            i -= 1

        return matches, unmatched_detections, unmatched_trackers

    def update(self, dets: np.ndarray, limit_x = 1000, max_age = 5):
        ms, uds, uts = self.match_dets(dets)

        current_trackers = []
        current_trackers_det = []

        for m in ms:
            t = self.current_det[m[1]]
            t.update(dets[m[0]])
            current_trackers.append(t)
            current_trackers_det.append(t)

        for u in uds:
            det = dets[u]
            t = VehicleTracker(det)
            current_trackers_det.append(t)

        current_trackers.sort(key=lambda t: t.id)

        for ut in uts:
            t = self.current_det[ut]
            x = t.ukf.x
            # offscreen
            if x[0] > limit_x or x[0] < 0:
                continue

            # too few frames, ignore
            if len(t.det_history) < 2:
                continue

            if t.predict_count < max_age:
                current_trackers.append(t)
                current_trackers_det.append(t)

        self.current = current_trackers
        self.current_det = current_trackers_det
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
