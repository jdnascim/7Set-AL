#!/bin/bash

declare -a models=("NGCN" "NAtt")
declare -a actl=("kmeans" "leiden-mci" "random")
declare -a emb=("clipsum", "clipcat")
declare -a train_val=((4,2) (10,6) (18,12) (90,30) ())

for model in "${models[@]}"; do
    for act in "${actl[@]}"; do
        python3 few_shot_karate_run.py --model $model --emb clipsum --actl $act --train_size 18 --val_size 12
    done
done