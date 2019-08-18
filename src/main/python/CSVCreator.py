import logging.config
import argparse
import os
import pandas as pd
import xml.etree.ElementTree as etree

logging.config.fileConfig(fname='../resources/logging.conf',
                          disable_existing_loggers=False)
log = logging.getLogger('csv_creator')


class CSVCreator(object):
    def __init__(self, input_path, output_path):
        self.annotation_files = [
            os.path.join(input_path, f) for f in os.listdir(input_path)
            if f.endswith('xml')
        ]
        self.output_path = output_path

    def run(self):
        result = pd.DataFrame()
        # import random
        # self.annotation_files = random.sample(self.annotation_files, 10)

        for annotation_file in self.annotation_files:
            xml_root = etree.parse(annotation_file).getroot()

            size_node = xml_root.find('size')

            for member in xml_root.findall('object'):
                bndbox = member.find('bndbox')

                result = result.append(
                    pd.DataFrame(
                        {
                            'filename':
                            os.path.basename(
                                annotation_file.replace('.xml', '.jpg')),
                            'width':
                            size_node.find('width').text,
                            'height':
                            size_node.find('height').text,
                            'xmin':
                            bndbox.find('xmin').text,
                            'ymin':
                            bndbox.find('ymin').text,
                            'xmax':
                            bndbox.find('xmax').text,
                            'ymax':
                            bndbox.find('ymax').text,
                            'class':
                            'bee'
                        },
                        index=[0]))

        result.to_csv(os.path.join(self.output_path, 'annotations.csv'),
                      index=False)


if __name__ == '__main__':
    log.info('CSVCreator - start')
    parser = argparse.ArgumentParser(description='CSVCreator')
    parser.add_argument('-i',
                        '--input_path',
                        help='path of the xml annotations',
                        required=True)
    parser.add_argument('-o',
                        '--output_path',
                        help='name of the output path',
                        required=True)

    args = parser.parse_args()
    log.info("Got the following args: {}".format(args))
    creator = CSVCreator(args.input_path, args.output_path)
    creator.run()

    log.info('CSVCreator - end')
