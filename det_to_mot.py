"""
Converts detections from a neural network into a mot file
without id assignment. Useful for some trackers implementations
"""

import os
import argparse
import pandas as pd 


def yolo_to_mot(csv_name: str, folder: str, width: int, height: int) -> None:
    files = [os.path.join(folder, f) for f in os.listdir(folder)]
    names=["frame", "id", "bb_left", "bb_top", 
            "bb_width", "bb_height", "conf", 
            "x", "y", "z"]
    all_data = []
    for file in files:
        frame = int(file.split("_")[-1].split(".")[0])
        det_df = pd.read_csv(file, sep=" ",
                         names=["class_id", "cx", "cy", "w", "h", "conf"])
        for _, row in det_df.iterrows():
            all_data.append({
                "frame": frame, "id": -1, 
                "bb_left": (row["cx"] - row["w"])*width,
                "bb_top": (row["cy"] - row["h"])*height, 
                "bb_width": row["w"] * width,
                "bb_height": row["h"] * height, 
                "conf": row["conf"], 
                "x": -1, "y": -1, "z": -1
            })

    pd.DataFrame(all_data).sort_values(by=["frame"]).to_csv(csv_name, header=False, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', help="folder where the detections are stored")
    parser.add_argument('-n', '--name', help="name of the generated cvs")
    parser.add_argument('-w', '--width', type=float, help="original image width")
    parser.add_argument('-t', '--height', type=float, help="original image height")
    args = parser.parse_args()
    yolo_to_mot(args.name, args.folder, args.width, args.height)

