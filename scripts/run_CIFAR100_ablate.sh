#!/bin/bash
SEEDS=( 42 )
NUM_RUNS=1
counter=0

for seed in "${SEEDS[@]}"; do

    cmd="CUDA_VISIBLE_DEVICES=0 python run.py -m models=trainPU_labelshift.yaml datamodule=random_split_module.yaml seed=${seed} dataset=CIFAR100 num_source_classes=75 fraction_ood_class=0.25"

    eval $cmd &
    sleep 10

    cmd="CUDA_VISIBLE_DEVICES=1 python run.py -m models=trainKPU.yaml datamodule=random_split_module.yaml seed=${seed} dataset=CIFAR100 num_source_classes=75 fraction_ood_class=0.25"

    eval $cmd &
    sleep 10

    cmd="CUDA_VISIBLE_DEVICES=2 python run.py -m models=sourceDiscriminator.yaml datamodule=random_split_module.yaml seed=${seed} dataset=CIFAR100 num_source_classes=75 fraction_ood_class=0.25"

    eval $cmd &
    sleep 10

    cmd="CUDA_VISIBLE_DEVICES=3 python run.py -m models=backpropODASaito.yaml datamodule=random_split_module.yaml seed=${seed} dataset=CIFAR100 learning_rate=0.001 num_source_classes=75 fraction_ood_class=0.25"

    eval $cmd &
    sleep 10

    counter=$((counter+1))
    if ! ((counter % NUM_RUNS)); then
        wait
    fi

done


for seed in "${SEEDS[@]}"; do

    cmd="CUDA_VISIBLE_DEVICES=0 python run.py -m models=trainPU_labelshift.yaml datamodule=random_split_module.yaml seed=${seed} dataset=CIFAR100 num_source_classes=65 fraction_ood_class=0.35"

    eval $cmd &
    sleep 10

    cmd="CUDA_VISIBLE_DEVICES=1 python run.py -m models=trainKPU.yaml datamodule=random_split_module.yaml seed=${seed} dataset=CIFAR100 num_source_classes=65 fraction_ood_class=0.35"

    eval $cmd &
    sleep 10

    cmd="CUDA_VISIBLE_DEVICES=2 python run.py -m models=sourceDiscriminator.yaml datamodule=random_split_module.yaml seed=${seed} dataset=CIFAR100 num_source_classes=65 fraction_ood_class=0.35"

    eval $cmd &
    sleep 10

    cmd="CUDA_VISIBLE_DEVICES=3 python run.py -m models=backpropODASaito.yaml datamodule=random_split_module.yaml seed=${seed} dataset=CIFAR100 learning_rate=0.001 num_source_classes=65 fraction_ood_class=0.35"

    eval $cmd &
    sleep 10

    counter=$((counter+1))
    if ! ((counter % NUM_RUNS)); then
        wait
    fi

done
