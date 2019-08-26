#!/bin/bash

python ../python/TensorflowRecordCreator.py -i ../../../data/resized/annotations_train.csv -o ../../../data/resized/train.record -ip ../../../data/resized/images
python ../python/TensorflowRecordCreator.py -i ../../../data/resized/annotations_test.csv -o ../../../data/resized/test.record -ip ../../../data/resized/images
