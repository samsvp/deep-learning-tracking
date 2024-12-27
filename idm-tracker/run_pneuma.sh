#!/bin/bash

N=$1
for W in $(seq 7 11); do
    python3 main.py -d ../mots/yolo-tiny/pNEUMA$N-tiny$W.mot --dir-x -1 -o output/pNEUMA$N-$W-tiny.mot -p
done
