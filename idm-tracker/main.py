import utils
import numpy as np
import pandas as pd
from idm_tracker import *
import draw
import cv2
import utils

np.random.seed(0)

def get_boxes(trackers: List[VehicleTracker]) -> np.ndarray:
    return np.array([t.get_boxes() for t in trackers])

def get_detections(df: pd.DataFrame, frame: int) -> np.ndarray:
    return df[df["frame"] == frame].values[:, 2:7] #type: ignore

#gt_mot = u.load_mot("10_0900_0930_D10_RM_mot.txt")
#p_mot = u.load_mot("sort/output/pNEUMA10_8-tiny.txt")
#acc_mot = u.load_mot("sort-acc/output/pNEUMA10_8-tiny.txt")
det_mot = utils.load_mot_road("../mots/yolo-tiny/cars-tiny-8.mot")

#frame1 = u.get_frame("pNEUMA10/", 1)
#u.view_frame(frame1, det_mot, 1)
# initialize all trackers
dets = get_detections(det_mot, 1)
trackers = Trackers(dets)

for i in range(2, 1908):
    print(i)
    dets = get_detections(det_mot, i)
    raw_dets = dets.copy()

    trackers.predict()
    trackers.update(dets, max_age=5)

    frame = draw.get_frame("../pNEUMA10/", i)
    frame_det = frame.copy()
    frame_raw_dets = frame.copy()
    draw.draw_all_rects(frame, get_boxes(trackers.current))
    draw.draw_all_rects(frame_det, get_boxes(trackers.current_det))
    #draw.draw_all_rects(frame_det, get_pred_boxes(trackers.current))
    draw.draw_all_rects(frame_raw_dets, [(i, *det) for i, det in enumerate(raw_dets)], True) #type: ignore

    cv2.imshow("frame", frame)
    cv2.imshow("frame det", frame_det)
    cv2.imshow("frame raw det", frame_raw_dets)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()

mot = trackers.to_mot()
with open("road.mot", 'w') as fp:
    fp.write(mot)

print("done")
