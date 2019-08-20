import logging.config
import argparse
import os
import pandas as pd
import xml.etree.ElementTree as etree

logging.config.fileConfig(fname='../resources/logging.conf', disable_existing_loggers=False)
log = logging.getLogger('csv_creator')


class CSVCreator(object):
    def __init__(self, input_path, output_path):
        self.annotation_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('xml')]
        self.output_path = output_path
        self.result = None

    def run(self):
        self.result = pd.DataFrame()
        # import random
        # self.annotation_files = random.sample(self.annotation_files, 10)

        for annotation_file in self.annotation_files:
            xml_root = etree.parse(annotation_file).getroot()
            size_node = xml_root.find('size')

            for member in xml_root.findall('object'):
                bndbox = member.find('bndbox')

                self.result = self.result.append(
                    pd.DataFrame(
                        {
                            'filename': os.path.basename(annotation_file.replace('.xml', '.jpg')),
                            'width': size_node.find('width').text,
                            'height': size_node.find('height').text,
                            'xmin': bndbox.find('xmin').text,
                            'ymin': bndbox.find('ymin').text,
                            'xmax': bndbox.find('xmax').text,
                            'ymax': bndbox.find('ymax').text,
                            'class': 'bee'
                        },
                        index=[0]))

        self.result.to_csv(os.path.join(self.output_path, 'annotations.csv'), index=False)

    def split_train_test(self):
        grouped = self.result.groupby('filename')
        train_size = int(len(self.result) * 0.8)
        # test_size = len(self.result) - train_size

        group_size = 0
        train_df = pd.DataFrame()
        for name, group in grouped:
            if group_size > train_size:
                break

            train_df = train_df.append(group)
            group_size += len(group)

        test_filenames = set(self.result['filename']).difference(set(train_df['filename']))
        test_df = self.result[self.result['filename'].isin(test_filenames)]
        train_df.to_csv(os.path.join(self.output_path, 'annotations_train.csv'), index=False)
        test_df.to_csv(os.path.join(self.output_path, 'annotations_test.csv'), index=False)
        log.info('split_train_test done')


if __name__ == '__main__':
    log.info('CSVCreator - start')
    parser = argparse.ArgumentParser(description='CSVCreator')
    parser.add_argument('-i', '--input_path', help='path of the xml annotations', required=True)
    parser.add_argument('-o', '--output_path', help='name of the output path', required=True)

    args = parser.parse_args()
    log.info("Got the following args: {}".format(args))
    creator = CSVCreator(args.input_path, args.output_path)
    creator.run()
    creator.split_train_test()

    log.info('CSVCreator - end')
