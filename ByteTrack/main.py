import cv2
import argparse
import numpy as np
import pandas as pd
from yolox.tracker.byte_tracker import BYTETracker

from dataclasses import dataclass 
from typing import *


@dataclass 
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

@dataclass 
class BB:
    x_left: int
    y_top: int 
    width: int
    height: int
    id: int


def get_random_color(id: int) -> Tuple[int, int, int]:
    return (id * 1512354 % 256, id * 231245 % 256, id * 5452356 % 256)


def draw_rect(frame: np.ndarray, bb: BB) -> None:
    color = get_random_color(bb.id)
    cv2.rectangle(
        frame, 
        (bb.x_left, bb.y_top), 
        (bb.x_left + bb.width, bb.y_top + bb.height), 
        color=color, thickness=2
    )
    cv2.putText(frame, str(bb.id), (bb.x_left, bb.y_top), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def main(mot_file: str, video_path: str, csv_name: str,
         skip_video: bool) -> None:
    df = pd.read_csv(mot_file,
                 names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"])

    byte_args = BYTETrackerArgs()
    tracker = BYTETracker(byte_args)
    cap = cv2.VideoCapture(video_path)
    frame_n = 1
    all_data = []
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            break

        dets = df[df["frame"]==frame_n][["bb_left", "bb_top", "bb_width", "bb_height", "conf"]]

        dets["bb_width"] += dets["bb_left"]
        dets["bb_height"] += dets["bb_top"]

        online_targets = tracker.update(output_results=dets.to_numpy(),
                       img_info=frame.shape,
                       img_size=frame.shape)
        tracks = []
        for track in online_targets:
            tlwh = track.tlwh
            bb = BB(
                int(tlwh[0] + tlwh[2]//2), 
                int(tlwh[1] + tlwh[3]//2), 
                int(tlwh[2]), 
                int(tlwh[3]), 
                track.track_id)
            tracks.append(bb)
            all_data.append({
                "frame": frame_n, "id": track.track_id, 
                "bb_left": tlwh[0],
                "bb_top": tlwh[1], 
                "bb_width": tlwh[2],
                "bb_height": tlwh[3], 
                "conf": track.score, 
                "x": -1, "y": -1, "z": -1
            })
        frame_n += 1
        
        if skip_video:
            if frame_n % 100 == 0 and frame_n != 0:
                print(f"{frame_n} frames processed")
            continue

        for track in tracks:
            draw_rect(frame, track)

        cv2.imshow('frame', frame)
        if cv2.waitKey(50) == ord('q'):
            break

    cap.release()
    if not skip_video:
        cv2.destroyAllWindows()
    
    pd.DataFrame(all_data).sort_values(by=["frame"]).to_csv(csv_name, header=False, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', help='Video to track')
    parser.add_argument('-f', '--filename', help='MOT file with detections')
    parser.add_argument('-n', '--name',
                        default="out.mot",
                        help='MOT output file with all tracks')
    parser.add_argument('-s', '--skip-video',
                        action="store_true",
                        help="Don't show opencv video")
    args = parser.parse_args()
    main(args.filename, args.video, args.name, args.skip_video)
