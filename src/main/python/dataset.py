import logging.config
import random
from typing import Dict

import albumentations as aug
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset

logging.config.fileConfig(fname="../resources/logging.conf", disable_existing_loggers=False)
log = logging.getLogger("bee_dataset")

# https://www.kaggle.com/pestipeti/pytorch-starter-fasterrcnn-train


class BeeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, do_augmentation: bool = True):
        super().__init__()

        self.df = BeeDataset.prepare_annotations(df)
        self.file_names = df["file_name"].unique()
        log.info(f"Got {len(self.file_names)} file names")
        random.seed(42)
        random.shuffle(self.file_names)

        self.image_dir = image_dir
        self.do_augmentation = do_augmentation

    def __getitem__(self, index: int):
        file_name = self.file_names[index]
        records = self.df[self.df["file_name"] == file_name]
        image = cv2.imread(f"{self.image_dir}/{file_name}", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        boxes = records[["x", "y", "w", "h"]].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((records.shape[0],), dtype=torch.int64)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index]),
            "area": area,
            "iscrowd": iscrowd,
        }

        if not np.array_equal(boxes, np.zeros(shape=(1, 4))):
            if self.do_augmentation and not np.array_equal(boxes, np.zeros(shape=(1, 4))):
                image = Augmentation(image, labels, target).result_image
            else:
                image = Augmentation(image, labels, target, only_resize=True).result_image
        else:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((1, 1), dtype=torch.int64),
                "image_id": torch.tensor([index]),
                "area": area,
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
            }
            image = Augmentation(image, labels, target, zero_bounding_box=True).result_image

        log.debug(f"got {len(target['boxes'])} boxes")
        log.debug(f"image shape after augmentation {image.shape}")
        return image, target, file_name

    def __len__(self) -> int:
        return self.file_names.shape[0]

    @staticmethod
    def prepare_annotations(df: pd.DataFrame):
        df["x"] = df["xmin"]
        df["y"] = df["ymin"]
        df["w"] = df["xmax"] - df["xmin"]
        df["h"] = df["ymax"] - df["ymin"]
        df.drop(columns=["width", "height", "xmin", "ymin", "xmax", "ymax", "class"], inplace=True)

        df["x"] = df["x"].astype(np.float)
        df["y"] = df["y"].astype(np.float)
        df["w"] = df["w"].astype(np.float)
        df["h"] = df["h"].astype(np.float)
        return df


class Augmentation(object):
    def __init__(
        self,
        image: np.ndarray,
        labels: torch.Tensor,
        target: Dict,
        new_width: int = 400,
        keep_ratio: bool = False,
        only_resize: bool = False,
        zero_bounding_box: bool = False,
    ):
        self.image = image
        self.labels = labels
        self.target = target
        self.new_width = new_width
        self.keep_ratio = keep_ratio
        self.only_resize = only_resize
        self.zero_bounding_box = zero_bounding_box

        self.new_height = self.calc_height()

        if zero_bounding_box:
            self.result_image = self.run_zero_bounding_box()
        else:
            self.transforms = self.build_transforms()
            self.result_image = self.run()

    def calc_height(self):
        new_height = self.new_width
        if self.keep_ratio:
            height, width, _ = self.image.shape
            ratio = self.new_width / width
            new_height = int(height * ratio)
        return new_height

    def build_transforms(self):
        if self.only_resize:
            transforms_list = [
                aug.Resize(self.new_height, self.new_width, p=1.0),
                ToTensorV2(p=1.0),
            ]
        else:
            transforms_list = [
                aug.Resize(self.new_height, self.new_width, p=1.0),
                aug.Flip(p=0.5),
                aug.RandomCrop(height=int(self.new_height * 0.8), width=int(self.new_width * 0.8), p=0.1),
                aug.CoarseDropout(max_holes=8, max_height=64, max_width=64, fill_value=0, p=0.1),
                aug.GaussNoise(p=0.1),
                aug.RandomBrightnessContrast(p=0.1),
                aug.RandomGamma(p=0.1),
                aug.ShiftScaleRotate(scale_limit=0.1, rotate_limit=45, p=0.25),
                ToTensorV2(p=1.0),
            ]
        transforms = aug.Compose(transforms_list, bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},)
        return transforms

    def run(self):
        boxes_result = []
        while not boxes_result:
            sample = {"image": self.image, "bboxes": self.target["boxes"], "labels": self.labels}
            sample = self.transforms(**sample)
            sample["bboxes"] = [x for x in sample["bboxes"] if x]
            boxes_result = sample["bboxes"]
            sample["bboxes"] = boxes_result
        image = sample["image"]
        self.target["boxes"] = torch.stack(tuple(map(torch.tensor, zip(*sample["bboxes"])))).permute(1, 0)
        self.target["boxes"] = self.target["boxes"].float()
        return image

    def run_zero_bounding_box(self):
        image = aug.Resize(self.new_height, self.new_width, p=1.0).apply(self.image)
        image = ToTensorV2(p=1.0).apply(image)
        return image


if __name__ == "__main__":
    df_train = pd.read_csv("../../../data/raw/annotations_train.csv")
    dataset_train = BeeDataset(df_train, "../../../data/raw/images")

    log.info(dataset_train[0])
