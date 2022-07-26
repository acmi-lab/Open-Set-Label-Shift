#!/bin/bash
GPU_IDS=( 0 1 2 3 )
SEEDS=( 42 1234 2011 )
NUM_RUNS=1
NUM_GPUS=${#GPU_IDS[@]}
counter=0

for seed in "${SEEDS[@]}"; do

    # gpu_idx=$((counter % $NUM_GPUS))
    # gpu_id=${GPU_IDS[$gpu_idx]}

    cmd="CUDA_VISIBLE_DEVICES=0 python run.py -m models=trainPU_labelshift.yaml datamodule=random_split_module.yaml seed=${seed} dataset=dermnet arch=Resnet50 num_source_classes=18 fraction_ood_class=0.2 max_epochs=80 batch_size=32 pretrained=True learning_rate=0.0001 separate=True"

    eval $cmd &
    sleep 10

    cmd="CUDA_VISIBLE_DEVICES=1 python run.py -m models=trainKPU.yaml datamodule=random_split_module.yaml seed=${seed} dataset=dermnet arch=Resnet50 num_source_classes=18 fraction_ood_class=0.2 max_epochs=80 batch_size=32 pretrained=True learning_rate=0.0001"

    eval $cmd &
    sleep 10

    cmd="CUDA_VISIBLE_DEVICES=2 python run.py -m models=sourceDiscriminator.yaml datamodule=random_split_module.yaml seed=${seed} dataset=dermnet arch=Resnet50 num_source_classes=18 fraction_ood_class=0.2 max_epochs=80 batch_size=32 pretrained=True learning_rate=0.0001"

    eval $cmd &
    sleep 10
    

    cmd="CUDA_VISIBLE_DEVICES=3 python run.py -m models=backpropODASaito.yaml datamodule=random_split_module.yaml seed=${seed} dataset=dermnet arch=Resnet50 num_source_classes=18 fraction_ood_class=0.2 max_epochs=80 batch_size=32 pretrained=True learning_rate=0.0001"

    eval $cmd &
    sleep 10

    counter=$((counter+1))
    if ! ((counter % NUM_RUNS)); then
        wait
    fi

done
