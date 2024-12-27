#!/bin/bash

VIDEOS=(2 6 9 11 13 15)
for N in ${VIDEOS[@]}; do
    for W in $(seq 7 11); do
        echo "Running video $N, network $W"
        python3 main.py -f ../mots/yolo-tiny/pNEUMA$N-tiny$W.mot -n output/pNEUMA$N-$W-tiny.mot -s -v "../pNEUMA$N.mp4"
    done
done

