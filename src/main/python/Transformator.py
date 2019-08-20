import argparse
import logging.config
import cv2
import numpy as np
import xml.etree.ElementTree as etree
import os

logging.config.fileConfig(fname='../resources/logging.conf', disable_existing_loggers=False)
log = logging.getLogger('transformator')


class Transformator(object):
    def __init__(self, input_path, output_path, annotation_path, width):
        self.output_path = output_path
        self.output_annotation_path = os.path.join(output_path, 'annotations')
        self.output_images_path = os.path.join(output_path, 'images')
        for p in [self.output_path, self.output_annotation_path, self.output_images_path]:
            if not os.path.exists(p):
                os.makedirs(p)

        log.info('Got this input_path: {}'.format(input_path))
        self.image_files = [f for f in os.listdir(input_path) if f.endswith('jpg')]
        self.annotation_files = [f for f in os.listdir(annotation_path) if f.endswith('xml')]
        self.input_path = input_path
        self.annotation_path = annotation_path
        self.width = width

    def resize(self):
        for image_file in self.image_files:
            annotation_file = image_file.replace('.jpg', '.xml')

            if annotation_file in self.annotation_files:
                log.info('resizing image {}'.format(image_file))
                dim, ratio = Transformator.resize_image(os.path.join(self.input_path, image_file),
                                                        os.path.join(self.output_images_path, image_file), self.width)

                log.info('resizing annotation file {}'.format(annotation_file))
                Transformator.resize_annotation(os.path.join(self.annotation_path, annotation_file),
                                                os.path.join(self.output_annotation_path, annotation_file),
                                                os.path.join(self.output_images_path, image_file), dim, ratio)

    @staticmethod
    def resize_image(input_image, output_image, new_width):
        image = cv2.imread(input_image)
        height, width, _ = image.shape
        ratio = new_width / width
        dim = (new_width, int(height * ratio))

        new_image = cv2.resize(image, dim)
        cv2.imwrite(output_image, new_image)
        return dim, ratio

    @staticmethod
    def resize_annotation(input_file, output_file, image_name, dim, ratio):
        xml_root = etree.parse(input_file).getroot()

        size_node = xml_root.find('size')
        size_node.find('width').text = str(dim[0])
        size_node.find('height').text = str(dim[1])

        xml_root.find('folder').text = 'resized'
        xml_root.find('path').text = image_name.replace("\\", "/")

        for member in xml_root.findall('object'):
            bndbox = member.find('bndbox')

            xmin = bndbox.find('xmin')
            ymin = bndbox.find('ymin')
            xmax = bndbox.find('xmax')
            ymax = bndbox.find('ymax')

            xmin.text = str(int(np.round(int(xmin.text) * ratio)))
            ymin.text = str(int(np.round(int(ymin.text) * ratio)))
            xmax.text = str(int(np.round(int(xmax.text) * ratio)))
            ymax.text = str(int(np.round(int(ymax.text) * ratio)))

        tree = etree.ElementTree(xml_root)
        tree.write(output_file)


if __name__ == '__main__':
    log.info('Transformator - start')
    parser = argparse.ArgumentParser(description='Transformator')
    parser.add_argument('-i', '--input_path', help='name of the input path', required=True)
    parser.add_argument('-o', '--output_path', help='name of the output folder', required=True)
    parser.add_argument('-a', '--annotation_path', help='name of the annotations folder', required=True)
    parser.add_argument('-w', '--width', help='width for the new size', required=False, default=600)

    args = parser.parse_args()
    log.info("Got the following args: {}".format(args))

    trafo = Transformator(args.input_path, args.output_path, args.annotation_path, args.width)
    trafo.resize()

    log.info('Transformator - end')
