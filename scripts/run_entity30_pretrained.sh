#!/bin/bash
SEEDS=( 42 )
# SEEDS=( 42 1234 2011 )
NUM_RUNS=3

for seed in "${SEEDS[@]}"; do

    cmd="CUDA_VISIBLE_DEVICES=2 python run.py -m models=trainPU_labelshift.yaml datamodule=random_split_module.yaml seed=${seed} dataset=entity30 arch=Resnet50 num_source_classes=24 fraction_ood_class=0.2 max_epochs=100 batch_size=64 pretrained=True learning_rate=0.0001 weight_decay=0.00"

    eval $cmd &
    sleep 10

    cmd="CUDA_VISIBLE_DEVICES=2 python run.py -m models=trainKPU.yaml datamodule=random_split_module.yaml seed=${seed} dataset=entity30 arch=Resnet50 num_source_classes=24 fraction_ood_class=0.2 max_epochs=100 batch_size=64 pretrained=True learning_rate=0.0001 weight_decay=0.00"

    eval $cmd &
    sleep 10

    cmd="CUDA_VISIBLE_DEVICES=2 python run.py -m models=sourceDiscriminator.yaml datamodule=random_split_module.yaml seed=${seed} dataset=entity30 arch=Resnet50 num_source_classes=24 fraction_ood_class=0.2 max_epochs=100 batch_size=64 pretrained=True learning_rate=0.0001 weight_decay=0.00"

    eval $cmd &
    sleep 10

    cmd="CUDA_VISIBLE_DEVICES=2 python run.py -m models=backpropODASaito.yaml datamodule=random_split_module.yaml seed=${seed} dataset=entity30 arch=Resnet50 num_source_classes=24 fraction_ood_class=0.2 max_epochs=100 batch_size=64 pretrained=True learning_rate=0.0001 weight_decay=0.00"

    eval $cmd &
    sleep 10

    counter=$((counter+1))
    if ! ((counter % NUM_RUNS)); then
        wait
    fi

done