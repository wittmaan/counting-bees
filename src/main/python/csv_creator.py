import argparse
import logging.config
import xml.etree.ElementTree as ET
from os import path, listdir

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.config.fileConfig(fname="../resources/logging.conf", disable_existing_loggers=False)
log = logging.getLogger("csv_creator")


class CSVCreator(object):
    def __init__(self, input_path, output_path):
        annotations_path = path.join(input_path, "annotations")
        self.images_path = path.join(input_path, "images")

        self.annotation_files = [path.join(annotations_path, f) for f in listdir(annotations_path) if f.endswith("xml")]
        self.image_files = [path.join(self.images_path, f) for f in listdir(self.images_path)]
        self.output_path = output_path
        self.df = None

    def run(self):
        self.match_annotations_to_images()
        self.find_images_without_annotations()

    def find_images_without_annotations(self):
        annotation_files_with_image_extension = [
            _.replace(".xml", ".jpg") for _ in [path.basename(_) for _ in self.annotation_files]
        ]
        image_files = [path.basename(_) for _ in self.image_files]
        image_files_without_annotation = sorted(np.setdiff1d(image_files, annotation_files_with_image_extension))
        df = pd.DataFrame(image_files_without_annotation, columns=["file_name"])
        df.to_csv(path.join(self.output_path, "files_without_annotations.csv"), index=False)

    def match_annotations_to_images(self):
        self.df = pd.DataFrame()
        for annotation_file in self.annotation_files:
            xml_root = ET.parse(annotation_file).getroot()
            size_node = xml_root.find("size")

            for member in xml_root.findall("object"):
                bndbox = member.find("bndbox")

                self.df = self.df.append(
                    pd.DataFrame(
                        {
                            "file_name": path.basename(annotation_file.replace(".xml", ".jpg")),
                            "width": size_node.find("width").text,
                            "height": size_node.find("height").text,
                            "xmin": bndbox.find("xmin").text,
                            "ymin": bndbox.find("ymin").text,
                            "xmax": bndbox.find("xmax").text,
                            "ymax": bndbox.find("ymax").text,
                            "class": "bee",
                        },
                        index=[0],
                    )
                )

    def split_train_valid(self):
        file_names = list(self.df["file_name"])
        # detect files which are from single recorded images
        image_file_names = [_ for _ in file_names if "Image" in _]
        file_names = [_ for _ in file_names if _ not in image_file_names]

        train_file_names, valid_file_names = train_test_split(list(set(file_names)), test_size=0.2, random_state=0)
        # put all single recorded images to the train data set
        train_file_names = set(image_file_names + train_file_names)

        df_train = self.df[self.df["file_name"].isin(train_file_names)]
        df_valid = self.df[self.df["file_name"].isin(valid_file_names)]

        df_train.to_csv(path.join(self.output_path, "annotations_train.csv"), index=False)
        df_valid.to_csv(path.join(self.output_path, "annotations_valid.csv"), index=False)
        log.info("split_train_valid done")


if __name__ == "__main__":
    log.info("CSVCreator - start")
    parser = argparse.ArgumentParser(description="CSVCreator")
    parser.add_argument("-i", "--input_path", help="path of the xml annotations", required=True)
    parser.add_argument("-o", "--output_path", help="name of the output path", required=True)

    args = parser.parse_args()
    log.info("Got the following args: {}".format(args))
    creator = CSVCreator(args.input_path, args.output_path)
    creator.run()
    creator.split_train_valid()

    log.info("CSVCreator - end")
