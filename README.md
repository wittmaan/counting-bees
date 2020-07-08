# counting-bees

## setup

### tensorflow

`conda create --name tensorflow python=3.6`

`pip install tensorflow-gpu`
`pip install --upgrade tf_slim`
 
`pip install pandas`

`pip install opencv-python`

`pip install pillow`

`pip install Cython`

`pip install pycocotools`

`pip install black`

### object detection api

`git clone https://github.com/tensorflow/models.git`


`sudo apt install protobuf-compiler`

in models/research run

`protoc object_detection/protos/*.proto --python_out=.`


follow the instructions from here 

`https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md`

additionally add `PYTHONPATH` variable to `PATH` variable.

run `(tensorflow-gpu) c:\Users\a.wittmann\Documents\misc\models\research>python setup.py install`


### google cloud

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_cloud.md

pip3 install --upgrade pip

#### storage

create storage `counting-bees-data` and in there create the following folders:
- `resized`: train.record and test.record should be uploaded there
- `pretrained`: put there the pretrained models from tensorflow model zoo
- `training`: to store any output files from the training, the pipeline.config file to run the training is also stored there.
- `evaluation`: to store any output files from evaluation 

#### command line

run 

`wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz`

`tar -zxvf faster_rcnn_resnet101_coco_2018_01_28.tar.gz`

`gsutil cp -r faster_rcnn_resnet101_coco_2018_01_28/ gs://counting-bees-data/pretrained`


run `git clone https://github.com/tensorflow/models.git` to get the tensorflow models repo

in models folder run `protoc object_detection/protos/*.proto --python_out=.`

run `pip3 install --user --upgrade tensorflow`


tensorboard --logdir=$MODEL_DIR --host=localhost --port=8080
