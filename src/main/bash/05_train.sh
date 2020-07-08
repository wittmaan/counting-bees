#!/bin/bash

export PYTHONPATH=$PYTHONPATH:../../../../models/research:../../../../models/research/slim

python ../python/train.py \
    --logtostderr \
    --train_dir=../../../data/training/faster_rcnn_resnet101 \
    --pipeline_config_path=../../../conf/pipeline.config

#MODEL_DIR=counting-bees-data/training/faster_rcnn_resnet101
#PIPELINE_CONFIG_PATH=counting-bees-data/training/faster_rcnn_resnet101/pipeline.config
#PATH_TO_LOCAL_YAML_FILE=/home/andreas_wittmann/cloud.yml

#cd /home/andreas_wittmann/models/research

#gcloud ai-platform jobs submit training object_detection_`date +%m_%d_%Y_%H_%M_%S` \
#    --runtime-version 1.12 \
#    --job-dir=gs://${MODEL_DIR} \
#    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
#    --module-name object_detection.model_main \
#    --region europe-west1 \
#    --config ${PATH_TO_LOCAL_YAML_FILE} \
#    -- \
#    --model_dir=gs://${MODEL_DIR} \
#    --pipeline_config_path=gs://${PIPELINE_CONFIG_PATH}
