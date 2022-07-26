#!/bin/bash
SEEDS=( 42 )
counter=0

for seed in "${SEEDS[@]}"; do

    cmd="CUDA_VISIBLE_DEVICES=1 python run.py -m models=trainPU_labelshift.yaml datamodule=random_split_module.yaml seed=${seed} dataset=CIFAR100 num_source_classes=85 fraction_ood_class=0.15 pretrained=True"

    eval $cmd &
    sleep 10

    cmd="CUDA_VISIBLE_DEVICES=1 python run.py -m models=trainKPU.yaml datamodule=random_split_module.yaml seed=${seed} dataset=CIFAR100 num_source_classes=85 fraction_ood_class=0.15 pretrained=True"

    eval $cmd &
    sleep 10

    cmd="CUDA_VISIBLE_DEVICES=1 python run.py -m models=sourceDiscriminator.yaml datamodule=random_split_module.yaml seed=${seed} dataset=CIFAR100 num_source_classes=85 fraction_ood_class=0.15 pretrained=True"

    eval $cmd &
    sleep 10

    cmd="CUDA_VISIBLE_DEVICES=1 python run.py -m models=backpropODASaito.yaml datamodule=random_split_module.yaml seed=${seed} dataset=CIFAR100 learning_rate=0.001 num_source_classes=85 fraction_ood_class=0.15 pretrained=True"

    eval $cmd &
    sleep 10

    counter=$((counter+1))
    if ! ((counter % NUM_RUNS)); then
        wait
    fi

done
