import argparse
import logging.config
from os import path

import cv2
import numpy as np
import pandas as pd

from detect import Detector

logging.config.fileConfig(fname="../resources/logging.conf", disable_existing_loggers=False)
log = logging.getLogger("image_detect")


class ImageDetector(Detector):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    def run(self):
        input_file = self.args.input_file
        image_files = pd.read_csv(input_file)
        image_files_list = [
            path.join(path.dirname(input_file), "images", _) for _ in image_files["file_name"].to_list()
        ]

        df = pd.DataFrame()
        for image_file in image_files_list:
            print(f"detection on {image_file}")
            df = df.append(self.detect_single_image(image_file))

        df = df.sort_values(by=["num_detections"])
        df.to_csv(path.join(path.dirname(input_file), "active_learning_result.csv"), index=False)

    def detect_single_image(self, input_file):
        image = cv2.imread(input_file)
        detections = self.detect(image)
        scores = detections[0]["scores"].cpu().numpy()
        least_scores = scores[np.argsort(scores)][:5]
        while len(least_scores) < 5:
            least_scores = np.append(least_scores, -1.0)

        metrics_dict = {
            "file_name": path.basename(input_file),
            "num_detections": len(scores),
            "mean": np.mean(scores) if len(scores) != 0 else -1,
            "std": np.std(scores) if len(scores) != 0 else -1,
            "max": np.max(scores) if len(scores) != 0 else -1,
            "least_val0": least_scores[0],
            "least_val1": least_scores[1],
            "least_val2": least_scores[2],
            "least_val3": least_scores[3],
            "least_val4": least_scores[4],
        }
        return pd.DataFrame(metrics_dict, index=[0])


if __name__ == "__main__":
    log.info("Detect - start")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="name of the input image file")
    parser.add_argument("--weights_path", help="path of the weights file")
    parsed_args = parser.parse_args()
    log.info("got the following args: {}".format(parsed_args))

    detector = ImageDetector(parsed_args)
    detector.run()

    log.info("Detect - end")
