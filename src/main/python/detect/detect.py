import argparse
import logging.config
from abc import ABCMeta, abstractmethod

import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms

logging.config.fileConfig(fname="../resources/logging.conf", disable_existing_loggers=False)
log = logging.getLogger("detect")


class Detector(object, metaclass=ABCMeta):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model, self.device = Detector.load_model(args.weights_path)

    @staticmethod
    def load_model(weights_path):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, box_detections_per_img=1000)
        num_classes = 2  # 1 class (bee) + background
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        model = model.to(device)

        model.load_state_dict(torch.load(weights_path))  # ['model'])
        model.eval()

        log.info(f"model from {weights_path} loaded")

        return model, device

    @abstractmethod
    def run(self):
        pass

    def count_detections(self, frame):
        detections = self.detect(frame)

        boxes = detections[0]["boxes"]
        scores = detections[0]["scores"]
        boxes = boxes[scores > 0.4]
        scores = scores[scores > 0.4]

        keep_idx = nms(boxes, scores, 0.70)
        boxes = boxes[keep_idx]

        for bbox in boxes:
            box = bbox.data.cpu().numpy()
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (220, 0, 0), 2)

        return len(boxes)

    def detect(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        img_transforms = transforms.Compose([transforms.ToTensor()])
        image_tensor = img_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            detections = self.model(image_tensor)
        return detections
