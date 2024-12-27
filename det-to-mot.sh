#!/bin/bash

VIDEOS=(2 6 9 11 13 15)
for N in ${VIDEOS[@]}; do
    for W in $(seq 7 10); do
        echo "pNEUMA $N, network $W"
        python3 det_to_mot.py -f pNEUMA/pNEUMA$N/pNEUMA$N-tiny$W/labels/ -n mots/yolo-tiny/pNEUMA$N-tiny$W.mot -w 1024 -t 525
    done
done
