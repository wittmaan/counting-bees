import logging.config
import argparse
import os
import cv2

logging.config.fileConfig(fname='../resources/logging.conf',
                          disable_existing_loggers=False)
log = logging.getLogger('frame_extractor')


class FrameExtractor(object):
    def __init__(self, input_file, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.output_path = output_path
        self.input_file = input_file
        self.cap = cv2.VideoCapture(input_file)

        log.info('input_file {} has framerate {}'.format(
            input_file, self.cap.get(cv2.CAP_PROP_FPS)))

    def run(self):
        frame_count = 0
        while True:
            success, image = self.cap.read()

            if not success:
                log.info(
                    'stopped extracting after {} frames'.format(frame_count))
                break

            filename = os.path.join(
                self.output_path,
                os.path.splitext(os.path.basename(self.input_file))[0] +
                '_frame{:08d}.jpg'.format(frame_count))
            frame_count += 1

            # extract the first and every 20th frame
            if frame_count == 1 or ((frame_count % 20) == 0):
                cv2.imwrite(filename, image)


if __name__ == '__main__':
    log.info('FrameExtractor - start')
    parser = argparse.ArgumentParser(description='FrameExtractor')
    parser.add_argument('-i',
                        '--input_file',
                        help='name including path of the input file',
                        required=True)
    parser.add_argument('-o',
                        '--output_path',
                        help='name of the output folder',
                        required=True)

    args = parser.parse_args()
    log.info("Got the following args: {}".format(args))
    extractor = FrameExtractor(args.input_file, args.output_path)
    extractor.run()

    log.info('FrameExtractor - end')
