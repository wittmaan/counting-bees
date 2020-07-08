#!/bin/bash

python ../python/tensorflow_record_creator.py -i ../../../data/resized/annotations_train.csv -o ../../../data/resized/train.record -ip ../../../data/resized/images
python ../python/tensorflow_record_creator.py -i ../../../data/resized/annotations_test.csv -o ../../../data/resized/test.record -ip ../../../data/resized/images
