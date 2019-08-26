#!/bin/bash

INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=../../../data/training/faster_rcnn_resnet101/pipeline.config
TRAINED_CKPT_PREFIX=../../../data/training/faster_rcnn_resnet101/model.ckpt-10384
EXPORT_DIR=../../../data/training/faster_rcnn_resnet101

python j:/models/research/object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}