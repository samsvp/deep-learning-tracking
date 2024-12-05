from tqdm import tqdm
import numpy as np
import argparse
import pandas as pd
from idm_tracker import *
import draw
import cv2

np.random.seed(0)


def load_mot(mot_file: str) -> pd.DataFrame:
    return pd.read_csv(
        mot_file,
        names=[
            "frame",
            "id",
            "bb_left",
            "bb_top",
            "bb_width",
            "bb_height",
            "conf",
            "x",
            "y",
            "z"]
    )


def load_mot_road(mot_file: str) -> pd.DataFrame:
    df = pd.read_csv(
        mot_file,
        names=[
            "frame",
            "id",
            "bb_left",
            "bb_top",
            "bb_width",
            "bb_height",
            "conf",
            "x",
            "y",
            "z"]
    )

    # we are only interested in the bottom part of the file
    bottom_df = df[df["bb_top"] > 350].copy(deep=True)

    return bottom_df  # type: ignore


def get_boxes(trackers: List[Vehicle]) -> np.ndarray:
    return np.array([t.get_boxes() for t in trackers])


def get_detections(df: pd.DataFrame, frame: int) -> np.ndarray:
    return df[df["frame"] == frame].values[:, 2:7]  # type: ignore


def get_pred_boxes(trackers: List[Vehicle]) -> np.ndarray:
    return np.array([
        t.get_pred_boxes() for t in trackers
        if len(t.pred_history) > 0
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--det-file",
        help="File with detections in the mot format, where all ids are -1")
    parser.add_argument(
        "--dir-x",
        help="Road direction in the X axis. 1 means that the vehicles move from left to right, -1 from right to left.",
        type=int,
        default=0)
    parser.add_argument(
        "--dir-y",
        help="Road direction in the Y axis. 1 means that the vehicles move from top to bottom, -1 from bottom to top.",
        type=int,
        default=0)
    parser.add_argument(
        "-o",
        "--out-file",
        help="File to save outputs",
        default="out.mot")
    parser.add_argument(
        "-s",
        "--show-tracking",
        help="Show real time tracking",
        action="store_true")
    parser.add_argument(
        "--images-dir",
        help="Images in the format {frame_n}.jpg, where frame_n is padded with 0s to have 5 digits.",
        required=False)
    parser.add_argument(
        "-p",
        "--progress-bar",
        help="Show progress bar",
        action="store_true")
    args = parser.parse_args()

    det_mot = load_mot(args.det_file)

    dets = get_detections(det_mot, 1)
    trackers = VehicleTrackers(dets)

    print(f"Running file {args.det_file}. Output: {args.out_file}")

    for i in tqdm(range(2, det_mot.frame.max()), disable=not args.progress_bar):
        dets = get_detections(det_mot, i)
        raw_dets = dets.copy()

        trackers.predict()
        trackers.update(dets, max_age=5)

        if args.show_tracking:
            frame = draw.get_frame(args.images_dir, i)
            frame_det = frame.copy()
            frame_raw_dets = frame.copy()
            draw.draw_all_rects(frame, get_boxes(trackers.current))

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) == ord('q'):
                break

    if args.show_tracking:
        cv2.destroyAllWindows()

    mot = trackers.to_mot()
    with open(args.out_file, 'w') as fp:
        fp.write(mot)

    print(f"{args.det_file} done")
