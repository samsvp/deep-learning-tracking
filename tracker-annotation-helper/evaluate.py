import argparse
import numpy as np
import pandas as pd
import motmetrics as mm

from typing import *


def motMetricsEnhancedCalculator(gt_source: str, t_source: str, print_results=False) -> \
        "pd.DataFrame | OrderedDict[str | Any, Any] | Dict":
    # load ground truth
    gt = np.loadtxt(gt_source, delimiter=',')

    # load tracking output
    t = np.loadtxt(t_source, delimiter=',')

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    # Max frame number maybe different for gt and t files
    for frame in range(1, int(gt[:, 0].max()) + 1):
        # select id, x, y, width, height for current frame
        # required format for distance calculation is X, Y, Width, Height \
        # We already have this format
        gt_dets = gt[gt[:, 0] == frame, 1:6]  # select all detections in gt
        t_dets = t[t[:, 0] == frame, 1:6]  # select all detections in t

        C = mm.distances.iou_matrix(gt_dets[:, 1:], t_dets[:, 1:],
                                    max_iou=0.5)  # format: gt, t

        # Call update once for per frame.
        # format: gt object ids, t object ids, distance
        acc.update(gt_dets[:, 0].astype('int').tolist(),
                   t_dets[:, 0].astype('int').tolist(), C)

    mh = mm.metrics.create()

    summary = mh.compute(acc, metrics=['num_frames', 'idf1', 'idp', 'idr',
                                       'recall', 'precision', 'num_objects',
                                       'mostly_tracked', 'partially_tracked',
                                       'mostly_lost', 'num_false_positives',
                                       'num_misses', 'num_switches',
                                       'num_fragmentations', 'mota', 'motp'
                                       ],
                         name='acc')

    if print_results:
        strsummary = mm.io.render_summary(
            summary,
            namemap={'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'recall': 'Recall', \
                    'precision': 'Precision', 'num_objects': 'Num Objects', \
                    'mostly_tracked': 'Mostly Tracked', 'partially_tracked': 'Partially Tracked', \
                    'mostly_lost': 'Mostly Lost', 'num_false_positives': 'FP', \
                    'num_misses': 'FN', 'num_switches': '# Switches', \
                    'num_fragmentations': 'Fragmentations', 'mota': 'MOTA', 'motp': 'MOTP',  \
                    }
        )
        print(strsummary)
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to calculate MOT dataset metrics")
    parser.add_argument('-g', '--ground_truth', type=str, help='Detections ground truth')
    parser.add_argument('-d', '--detections', type=str, help='Predicted Detections')
    args = parser.parse_args()
    motMetricsEnhancedCalculator(args.ground_truth, args.detections, print_results=True)