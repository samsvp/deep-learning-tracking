import os 
from det_to_mot import yolo_to_mot


detect_path = "yolov7/runs/detect/"
labels_folder = [(f, os.path.join(detect_path, f, "labels"))
    for f in os.listdir(detect_path)]

if not os.path.exists("mots"):
    os.makedirs("mots")
if not os.path.exists("mots/yolo-tiny"):
    os.makedirs("mots/yolo-tiny")

mots_names = []
for f, label_folder in labels_folder:
    print(f"Converting {f} to mot")
    csv_name = f"mots/yolo-tiny/{f}.mot"
    mots_names.append(csv_name)
    if f.startswith("cars"):
        yolo_to_mot(csv_name, label_folder, 1024, 526)
    elif f.startswith("luton"):
        yolo_to_mot(csv_name, label_folder, 1280, 720)
    elif f.startswith("city"):
        yolo_to_mot(csv_name, label_folder, 1280, 720)


