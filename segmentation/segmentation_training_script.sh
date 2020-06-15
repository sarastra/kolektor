#!/usr/bin/env bash

train_segmentation_net() {
    echo STARTED $@
    PYTHONHASHSEED=0 python -u segmentation_cli.py \
        --split_file=../../KolektorSDD-training-splits/split.pyb \
        --root_dir=../../KolektorSDD \
        --subset_number=0 \
        --which_samples=${1} \
        --divide_image_size_by=${2} \
        --kernel_size=${3} \
        --learning_rate=${4} \
        --epochs=${5} \
        --device=${6} \
        --results_folder=${7} \
        --pretrained_model=${8}
    echo FINISHED $@
}
