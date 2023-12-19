import cv2
import numpy as np
import pandas as pd
import argparse

from typing import *
from dataclasses import dataclass
from deep_sort_realtime.deepsort_tracker import DeepSort

@dataclass 
class BB:
    x_left: int
    y_top: int 
    width: int
    height: int


def get_random_color(id: int) -> Tuple[int, int, int]:
    return (id * 1512354 % 256, id * 231245 % 256, id * 5452356 % 256)


def draw_rect(frame: np.ndarray, bb: BB, id: int) -> None:
    color = get_random_color(id)
    cv2.rectangle(
        frame, 
        (bb.x_left, bb.y_top), 
        (bb.x_left + bb.width, bb.y_top + bb.height), 
        color=color, thickness=2
    )
    cv2.putText(frame, str(id), (bb.x_left, bb.y_top), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def main(mot_file: str, video_path: str, csv_name: str,
         skip_video: bool) -> None:
    df = pd.read_csv(mot_file, 
                 names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"])
    tracker = DeepSort(max_age=5)
    n = 1
    all_data = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        bbs = []
        for _, row in df[df["frame"]==n].iterrows():
            x = int(row['bb_left'])
            y = int(row['bb_top'])
            w = int(row['bb_width'])
            h = int(row['bb_height'])
            bbs.append(([x,y,w,h], row['conf'], 0))
        tracks = tracker.update_tracks(bbs, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = int(track.track_id)
            ltrb = track.to_ltrb()
            x = int(ltrb[0])
            y = int(ltrb[1])
            w = int(ltrb[2]-ltrb[0])
            h = int(ltrb[3]-ltrb[1])
            if not skip_video:
                draw_rect(frame, 
                          BB(x + w//2, y + h//2, w, h), 
                          track_id)
            
            all_data.append({
                "frame": n, "id": track_id, 
                "bb_left": x,
                "bb_top": y, 
                "bb_width": w,
                "bb_height": h, 
                "conf": track.det_conf, 
                "x": -1, "y": -1, "z": -1
            })
        n += 1

        if n % 100 == 0:
            print(f"Processed {n} frames")
        if skip_video:
            continue

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
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

