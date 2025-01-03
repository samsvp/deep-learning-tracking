"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import print_function
from kalman_filters import *
import argparse
import time
import glob
from skimage import io
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


np.random.seed(0)
KF_BASE = "base"
KF_VEL = "vel"
KF_ACC = "acc"
KF_ADP = "adp"

def linear_assignment(cost_matrix):
    try:
        import lap  # type: ignore
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
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
    return (o)


def associate_detections_to_trackers(
        detections, trackers_boxes, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers_boxes) == 0):
        return np.empty((0, 2), dtype=int), np.arange(
            len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers_boxes)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, _ in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, _ in enumerate(trackers_boxes):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(
        unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, kf_type, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.kf = KalmanBoxTracker
        if kf_type == KF_VEL or kf_type == KF_BASE:
            self.kf = KalmanBoxTracker
        elif kf_type == KF_ACC:
            self.kf = KalmanBoxTrackerAcc
        elif kf_type == KF_ADP:
            self.kf = KalmanBoxTrackerAdaptative
        else:
            raise Exception(f"Invalid kalman filter type. Options: {KF_BASE}, {KF_VEL}, {KF_ACC}, {KF_ADP}")

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, _ = associate_detections_to_trackers(
            dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = self.kf(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >=
                                                self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument(
        '--display',
        dest='display',
        help='Display online tracker output (slow) [False]',
        action='store_true')
    parser.add_argument(
        "--seq_path",
        help="Path to detections.",
        type=str,
        default='data')
    parser.add_argument(
        "--phase",
        help="Subdirectory in seq_path.",
        type=str,
        default='train')
    parser.add_argument(
        "--max_age",
        help="Maximum number of frames to keep alive a track without associated detections.",
        type=int, default=1)
    parser.add_argument(
        "--min_hits",
        help="Minimum number of associated detections before track is initialised.",
        type=int, default=3)
    kf_models = f"{KF_BASE}/{KF_VEL} (constant velocity model), {KF_ACC} (constant acceleration model), {KF_ADP} (adaptative model)"
    parser.add_argument(
        "--kalman-filter",
        help=f"Which kind of kalman filter to use. {kf_models}",
        default="base")
    parser.add_argument(
        "--iou_threshold",
        help="Minimum IOU for match.",
        type=float,
        default=0.3)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # all train
    args = parse_args()
    display = args.display
    phase = args.phase
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)  # used only for display
    if (display):
        if not os.path.exists('mot_benchmark'):
            print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
            exit()
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal')

    if not os.path.exists('output'):
        os.makedirs('output')
    pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
    for seq_dets_fn in glob.glob(pattern):
        mot_tracker = Sort(
            kf_type=args.kalman_filter,
            max_age=args.max_age,
            min_hits=args.min_hits,
            iou_threshold=args.iou_threshold)  # create instance of the SORT tracker
        seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
        seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]

        with open(os.path.join('output', f'{seq}-{args.kalman_filter}.txt'), 'w') as out_file:
            print("Processing %s." % (seq))
            for frame in range(int(seq_dets[:, 0].max())):
                frame += 1  # detection and frame numbers begin at 1
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                dets[:, 2:4] += dets[:, 0:2]
                total_frames += 1

                if (display):
                    fn = os.path.join(
                        'mot_benchmark',
                        phase,
                        seq,
                        'img1',
                        '%06d.jpg' %
                        (frame))
                    im = io.imread(fn)
                    ax1.imshow(im)  # type: ignore
                    plt.title(seq + ' Tracked Targets')

                start_time = time.time()
                trackers = mot_tracker.update(dets)
                cycle_time = time.time() - start_time
                total_time += cycle_time

                for d in trackers:
                    print(
                        '%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' %
                        (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]), file=out_file)
                    if (display):
                        d = d.astype(np.int32)
                        ax1.add_patch(patches.Rectangle( #type: ignore
                            (d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3, ec=colours[d[4] % 32, :]))

                if (display):
                    fig.canvas.flush_events()  # type: ignore
                    plt.draw()
                    ax1.cla()  # type: ignore

    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" %
          (total_time, total_frames, total_frames / total_time))

    if (display):
        print("Note: to get real runtime results run without the option: --display")
