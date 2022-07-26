#!/bin/bash
SEEDS=( 42 1234 2011 )
NUM_RUNS=1
counter=0

for seed in "${SEEDS[@]}"; do
    cmd="CUDA_VISIBLE_DEVICES=0 python run.py -m models=danceSaito.yaml datamodule=random_split_module.yaml seed=${seed} num_source_classes=9 fraction_ood_class=0.1 learning_rate=0.01 models.num_target_samples=23247"

    eval $cmd &
    sleep 10

    cmd="CUDA_VISIBLE_DEVICES=1 python run.py -m models=separate2Adapt.yaml datamodule=random_split_module.yaml seed=${seed} num_source_classes=9 fraction_ood_class=0.1 learning_rate=0.001"

    eval $cmd &
    sleep 10

    cmd="CUDA_VISIBLE_DEVICES=2 python run.py -m models=UANYou19.yaml datamodule=random_split_module.yaml seed=${seed} num_source_classes=9 fraction_ood_class=0.1 learning_rate=0.001 max_epochs=1"

    eval $cmd &
    sleep 10

    cmd="CUDA_VISIBLE_DEVICES=3 python run.py -m models=uncertaintyFu20.yaml datamodule=random_split_module.yaml seed=${seed} learning_rate=0.01 num_source_classes=9 fraction_ood_class=0.001"

    eval $cmd &
    sleep 10

    counter=$((counter+1))
    if ! ((counter % NUM_RUNS)); then
        wait
    fi

done
