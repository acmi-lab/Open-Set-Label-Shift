 #!/bin/bash
SEEDS=( 42 1234 2011 )
NUM_RUNS=3
counter=0

for seed in "${SEEDS[@]}"; do

    cmd="CUDA_VISIBLE_DEVICES=0 python run.py -m models=trainPU_labelshift.yaml datamodule=random_split_module.yaml seed=${seed} dataset=newsgroups arch=Model_20 num_source_classes=16 fraction_ood_class=0.2 max_epochs=120 batch_size=128 pretrained=False learning_rate=0.0001 separate=True"

    eval $cmd &
    sleep 10

    cmd="CUDA_VISIBLE_DEVICES=1 python run.py -m models=trainKPU.yaml datamodule=random_split_module.yaml seed=${seed} dataset=newsgroups arch=Model_20 num_source_classes=16 fraction_ood_class=0.2 max_epochs=120 batch_size=128 pretrained=False learning_rate=0.0001"

    eval $cmd &
    sleep 10

    cmd="CUDA_VISIBLE_DEVICES=2 python run.py -m models=sourceDiscriminator.yaml datamodule=random_split_module.yaml seed=${seed} dataset=newsgroups arch=Model_20 num_source_classes=16 fraction_ood_class=0.2 max_epochs=120 batch_size=128 pretrained=False learning_rate=0.0001"

    eval $cmd &
    sleep 10

    cmd="CUDA_VISIBLE_DEVICES=3 python run.py -m models=backpropODASaito.yaml datamodule=random_split_module.yaml seed=${seed} dataset=newsgroups arch=Model_20 num_source_classes=16 fraction_ood_class=0.2 max_epochs=120 batch_size=128 pretrained=False learning_rate=0.0001"

    eval $cmd &
    sleep 10

    counter=$((counter+1))
    if ! ((counter % NUM_RUNS)); then
        wait
    fi

done
