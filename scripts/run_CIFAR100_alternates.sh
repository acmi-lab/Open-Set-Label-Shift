#!/bin/bash
# GPU_IDS=( 0 1 2 3 )
SEEDS=( 42 )
NUM_RUNS=1
# NUM_GPUS=${#GPU_IDS[@]}
counter=0

for seed in "${SEEDS[@]}"; do

    # gpu_idx=$((counter % $NUM_GPUS))
    # gpu_id=${GPU_IDS[$gpu_idx]}
    cmd="CUDA_VISIBLE_DEVICES=1 python run.py -m models=trainPU_labelshift_ablate.yaml datamodule=random_split_module.yaml seed=${seed} dataset=CIFAR100 num_source_classes=85 fraction_ood_class=0.15"

    eval $cmd &
    sleep 10

    cmd="CUDA_VISIBLE_DEVICES=2 python run.py -m models=trainKPU_ablate.yaml datamodule=random_split_module.yaml seed=${seed} dataset=CIFAR100 num_source_classes=85 fraction_ood_class=0.15"

    eval $cmd &
    sleep 10


    counter=$((counter+1))
    if ! ((counter % NUM_RUNS)); then
        wait
    fi

done
