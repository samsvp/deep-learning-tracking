import os
import pandas as pd
from evaluate import motMetricsEnhancedCalculator


def calc_metrics(gt_file: str, prefix: str, bottom_half=False, right_half=False):
    print(f"Evaluating {prefix}")

    if not os.path.exists("mots/eval"):
        os.makedirs("mots/eval")

    ground_truth = pd.read_csv(gt_file,
                 names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"])
    if bottom_half:
        # look only at the bottom of the screen, as the top has been segmented
        # by the 16x16 split
        ground_truth = ground_truth[ground_truth["bb_top"] > 526//2]
    if right_half:
        # look only at the bottom of the screen, as the top has been segmented
        # by the 16x16 split
        ground_truth = ground_truth[ground_truth["bb_left"] > 400]

    for tracker in ["idm-tracker", "sort-acc", "deepsort", "ByteTrack"]:
        tracker_path = f"{tracker}/output"
        mot_names = [os.path.join(tracker_path, mot)
                     for mot in os.listdir(tracker_path)
                     if prefix in mot
        ]
        for mot in mot_names:
            print(f"Evaluating {mot}")
            out_prefix = tracker
            if tracker == "sort-acc":
                kind = mot.split(".")[0].split("-")[-1]
                out_prefix += f"-{kind}"

            csv_name = f"{out_prefix}-" + mot.split("/")[-1].replace(".txt", ".eval")
            df = pd.read_csv(mot,
                     names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"])

            if right_half:
                df = df[df["bb_left"] > 400]

            if bottom_half:
                df = df[df["bb_top"] > 526//2]

            metrics = motMetricsEnhancedCalculator(ground_truth.to_numpy(), df.to_numpy(), True)
            metrics.to_csv(os.path.join("mots/eval", csv_name))


calc_metrics("10_0900_0930_D10_RM_mot.txt", "pNEUMA10", True)
calc_metrics("2_0900_0930_D10_RM_mot.txt", "pNEUMA2", False, True)
for i in [6, 9, 11, 13, 15]:
    calc_metrics(f"{i}_0900_0930_D10_RM_mot.txt", f"pNEUMA{i}")
