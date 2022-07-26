#!/bin/bash
SEEDS=( 42 )
NUM_RUNS=1
counter=0

for seed in "${SEEDS[@]}"; do
    cmd="CUDA_VISIBLE_DEVICES=0  python run.py -m models=danceSaito.yaml datamodule=random_split_module.yaml seed=${seed} dataset=CIFAR100 num_source_classes=85 fraction_ood_class=0.15 models.num_target_samples=25255 learning_rate=0.01"

    eval $cmd &
    sleep 10

    cmd="CUDA_VISIBLE_DEVICES=1 python run.py -m models=separate2Adapt.yaml datamodule=random_split_module.yaml  seed=${seed} dataset=CIFAR100 num_source_classes=85 fraction_ood_class=0.15 learning_rate=0.001"

    eval $cmd &
    sleep 10

    cmd="CUDA_VISIBLE_DEVICES=2 python run.py -m models=UANYou19.yaml datamodule=random_split_module.yaml seed=${seed} dataset=CIFAR100 num_source_classes=85 fraction_ood_class=0.15 learning_rate=0.001"

    eval $cmd &
    sleep 10

    cmd="CUDA_VISIBLE_DEVICES=3 python run.py -m models=uncertaintyFu20.yaml datamodule=random_split_module.yaml  seed=${seed} dataset=CIFAR100 learning_rate=0.01 num_source_classes=85 fraction_ood_class=0.15 "

    eval $cmd &
    sleep 10

    counter=$((counter+1))
    if ! ((counter % NUM_RUNS)); then
        wait
    fi

done
