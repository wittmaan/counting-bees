import logging.config
import argparse
import os
import pandas as pd
import tensorflow as tf
from object_detection.utils import dataset_util
import io
import PIL.Image

logging.config.fileConfig(fname='../resources/logging.conf',
                          disable_existing_loggers=False)
log = logging.getLogger('tensorflow_record_creator')


class TensorflowRecordCreator(object):
    def __init__(self, input_file, output_file, images_path):
        self.annotations = pd.read_csv(input_file)
        self.output_file = output_file
        self.images_path = images_path

    def run(self):
        writer = tf.python_io.TFRecordWriter(self.output_file)
        for group in self.annotations.groupby('filename'):
            tf_example = self.create_tf_example(group, self.images_path)
            writer.write(tf_example.SerializeToString())
        writer.close()

    @staticmethod
    def create_tf_example(grouped, images_path):
        filename, group = grouped

        with tf.gfile.GFile(os.path.join(images_path, filename), 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)
        width, height = image.size

        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes_text = []
        classes = []

        for idx, val in group.iterrows():
            xmin.append(float(val['xmin']) / float(val['width']))
            ymin.append(float(val['ymin']) / float(val['height']))
            xmax.append(float(val['xmax']) / float(val['width']))
            ymax.append(float(val['ymax']) / float(val['height']))
            classes_text.append(val['class'].encode('utf8'))
            classes.append(1)  # TODO: use label_map_dict

        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image/height':
                dataset_util.int64_feature(height),
                'image/width':
                dataset_util.int64_feature(width),
                'image/filename':
                dataset_util.bytes_feature(filename.encode('utf8')),
                'image/source_id':
                dataset_util.bytes_feature(filename.encode('utf8')),
                'image/encoded':
                dataset_util.bytes_feature(encoded_jpg),
                'image/format':
                dataset_util.bytes_feature('jpeg'.encode('utf8')),
                'image/object/bbox/xmin':
                dataset_util.float_list_feature(xmin),
                'image/object/bbox/xmax':
                dataset_util.float_list_feature(xmax),
                'image/object/bbox/ymin':
                dataset_util.float_list_feature(ymin),
                'image/object/bbox/ymax':
                dataset_util.float_list_feature(ymax),
                'image/object/class/text':
                dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label':
                dataset_util.int64_list_feature(classes),
            }))
        return example


if __name__ == '__main__':
    log.info('TensorflowRecordCreator - start')
    parser = argparse.ArgumentParser(description='CSVCreator')
    parser.add_argument('-i',
                        '--input_file',
                        help='name of the input file',
                        required=True)
    parser.add_argument('-o',
                        '--output_file',
                        help='name of the output file',
                        required=True)
    parser.add_argument('-ip',
                        '--images_path',
                        help='path name of the images',
                        required=True)

    args = parser.parse_args()
    log.info("Got the following args: {}".format(args))
    creator = TensorflowRecordCreator(args.input_file, args.output_file,
                                      args.images_path)
    creator.run()

    log.info('TensorflowRecordCreator - end')
