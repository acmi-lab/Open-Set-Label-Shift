#!/bin/bash
SEEDS=( 42 )
counter=0
NUM_RUNS=1

for seed in "${SEEDS[@]}"; do

    cmd="CUDA_VISIBLE_DEVICES=3 python run.py -m models=trainPU_labelshift_ablate.yaml datamodule=random_split_module.yaml separate=True seed=${seed} num_source_classes=9 fraction_ood_class=0.1 learning_rate=0.0001"

    eval $cmd &
    sleep 10

    cmd="CUDA_VISIBLE_DEVICES=2 python run.py -m models=trainKPU_ablate.yaml datamodule=random_split_module.yaml seed=${seed} num_source_classes=9 fraction_ood_class=0.1 learning_rate=0.0001"

    eval $cmd &
    sleep 10


    counter=$((counter+1))
    if ! ((counter % NUM_RUNS)); then
        wait
    fi

done
