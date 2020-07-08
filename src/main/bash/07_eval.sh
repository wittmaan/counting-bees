#!/bin/bash

export PYTHONPATH=$PYTHONPATH:../../../../models/research:../../../../models/research/slim

python ../python/eval.py \
    --logtostderr \
    --pipeline_config_path=../../../data/training/faster_rcnn_resnet101/pipeline.config \
    --checkpoint_dir=../../../data/training/faster_rcnn_resnet101 \
    --eval_dir=../../../data/eval/faster_rcnn_resnet101
