import cv2
import argparse
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import *


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


def main(mot_file: str, video_path: str, save: bool,
         name: str) -> None:
    df = pd.read_csv(mot_file,
                 names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"])
    n = 1
    print(video_path)
    cap = cv2.VideoCapture(video_path)
    if save:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(name, fourcc, 20.0, (width, height))
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        for _, row in df[df["frame"]==n].iterrows():
            id = int(row['id'])
            x = int(row['bb_left'])
            y = int(row['bb_top'])
            w = int(row['bb_width'])
            h = int(row['bb_height'])
            bb = BB(x + w//2, y + h//2, w, h)
            draw_rect(frame, bb, id)
        n += 1

        if save:
            out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(100) == ord('q'):
            break

    cap.release()
    if save:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mot-file')
    parser.add_argument('-v', '--video',
                        default='pneuma10.mp4',
                        help="path of the video to be player")
    parser.add_argument('--save', action="store_true",
                        help="save the video with the bounding boxes")
    parser.add_argument('-n', '--name',
                        help="name to save the video to")
    args = parser.parse_args()
    main(args.mot_file, args.video, args.save, args.name)


