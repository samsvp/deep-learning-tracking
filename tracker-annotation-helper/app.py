import os
import re
import csv
import argparse
from PIL import Image
from flask import Flask, render_template, send_file, request, jsonify

from typing import *


app = Flask(__name__)

frame = -1
image_files = ['sample_images/frame_1.jpg',
               'sample_images/frame_271.jpg', 'sample_images/frame_272.jpg']
mot_data = {}

# Define the route for the home page


@app.route('/')
def home():
    # Load the background image
    img_filename = image_files[frame]
    bg_image = Image.open(img_filename)

    # Pass the image dimensions to the template
    bg_width, bg_height = bg_image.size

    # Render the template with the image dimensions
    return render_template('index.html',
                           frame_count=len(image_files),
                           img_filename=img_filename,
                           bg_width=bg_width, bg_height=bg_height)


@app.route('/load_data')
def load_data():
    return jsonify(mot_data)


@app.route('/get_current_frame')
def get_current_frame():
    return jsonify({"frame": frame})


@app.route('/get_image', methods=["POST"])
def get_image():
    global frame

    data = request.get_json()
    frame = data["frame"]

    if frame < 0:
        frame = 0
    elif frame >= len(image_files):
        frame = len(image_files) - 1

    return send_file(image_files[frame], mimetype='image/jpeg')


@app.route('/save_square', methods=["POST"])
def save_squares():
    try:
        data = request.get_json()
        squares = data["squares"]
        print(squares)
        with open(save_filename, "w") as f:
            f.write(squares)
        return jsonify({
            "status": "success",
            "msg": f"Frame {frame} data added"
        })
    except Exception as e:
        return jsonify({
            "status": "failure",
            "msg": str(e)
        })


def find_object_with_id(frame: int, id: int) -> Dict[str, float]:
    for obj in mot_data[frame]:
        if obj["id"] == id:
            return obj
    return {}


def load_mot_data(filename: str):
    mot_data = {}

    with open(filename) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            frame, id, bb_left, bb_top, bb_width, bb_height, _, _, _, _ = map(
                float, row)
            frame = int(frame - 1)

            if frame not in mot_data:
                mot_data[frame] = []

            mot_data[frame].append({
                "id": id, "bb_left": bb_left, "bb_top": bb_top,
                "bb_width": bb_width, "bb_height": bb_height})

    return mot_data


def get_image_filepaths(folderpath: str) -> List[str]:

    def natural_keys(text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [int(c) if c.isdigit() else c
                for c in re.split(r'(\d+)', text)]
    filenames = [os.path.join(folderpath, filename)
                 for filename in os.listdir(folderpath)]

    filenames.sort(key=natural_keys)
    return filenames


if __name__ == '__main__':
    # create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description="Flask app that lets the user fix partial tracking annotations")
    parser.add_argument('-m', '--mot_file', type=str,
                        help='Name of the file with the partial detections (MOT format)')
    parser.add_argument('-i', '--images', type=str,
                        help="Folder containing the images related to the MOT file. The images must be \
                            in the format *_d.{png/jpg}")
    parser.add_argument('-n', '--save_name', type=str, default="mot.txt",
                        help='Name to save the mot file')
    args = parser.parse_args()

    if args.mot_file is not None:
        mot_data = load_mot_data(args.mot_file)

    image_files = get_image_filepaths(args.images)
    save_filename = args.save_name

    app.run(debug=True)
