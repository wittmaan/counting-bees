import argparse
import logging.config

import cv2
import numpy as np

from detect import Detector

logging.config.fileConfig(fname="../resources/logging.conf", disable_existing_loggers=False)
log = logging.getLogger("video_detect")


class VideoDetector(Detector):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    def run(self):
        cap = cv2.VideoCapture(self.args.input_file)
        counted_list = []

        while True:
            ret, frame = cap.read()

            bees_counted = self.count_detections(frame)
            counted_list.append(bees_counted)
            counted_list = counted_list[-20:]

            cv2.putText(
                frame,
                f"detected {bees_counted} bees",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"moving average detected {np.mean(counted_list)} bees",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("frame", frame)

            if not ret or cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    log.info("Detect - start")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="name of the input video file")
    parser.add_argument("--weights_path", help="path of the weights file")
    parsed_args = parser.parse_args()
    log.info("got the following args: {}".format(parsed_args))

    detector = VideoDetector(parsed_args)
    detector.run()

    log.info("Detect - end")
