#%%
import utils as u
import numpy as np
import filters as f

np.random.seed(0)

class RoadTracker():
    def __init__(self, dets: np.ndarray):
        self.trackers = []
        for d in dets:
            tracker = f.KalmanBoxTracker(f.convert_det_bbox(d))
            self.trackers.append(tracker)

    def update(self, dets: np.ndarray):
        pass


gt_mot = u.load_mot("10_0900_0930_D10_RM_mot.txt")
p_mot = u.load_mot("sort/output/pNEUMA10_8-tiny.txt")
det_mot = u.load_mot("mots/yolo-tiny/cars-tiny-8.mot")
acc_mot = u.load_mot("sort-acc/output/pNEUMA10_8-tiny.txt")

frame1 = u.get_frame("pNEUMA10/", 1)
u.view_frame(frame1, gt_mot, 1)


# %%
df = u.load_mot("mots/yolo-tiny/cars-tiny-8.mot")
u.draw_boxes_plt(df, 1, frame1.shape)
rt = RoadTracker(df.values[:, 2:6])
# %%
