#!/bin/bash

run() {
    FILENAME="$1"
    OUTNAME="output/$(echo $FILENAME | rev | cut -d'/' -f1 | rev | cut -d. -f1).txt"
    python3 main.py -d $FILENAME --dir-x 1 -o $OUTNAME
}

export -f run

FILES=$(ls ../mots/yolo-tiny/cars*)
printf '%s\n' $FILES | parallel -j4 run {1}
