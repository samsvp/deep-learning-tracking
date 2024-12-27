#!/bin/bash

VIDEOS=(2 6 9 11 13 15)
for N in ${VIDEOS[@]}; do
    for W in $(seq 7 10); do
        FOLDER="data/val/pNEUMA${N}_${W}-tiny/det"
        echo "creating folder $FOLDER"
        mkdir -p "$FOLDER"
        cp "../mots/yolo-tiny/pNEUMA${N}-tiny$W.mot" $FOLDER/det.txt
    done
done
