import logging
from os import path
from typing import Set

import numpy as np
import torch
from torchvision.ops import box_iou

log = logging.getLogger("utils")


def collate_fn(batch):
    return tuple(zip(*batch))


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# https://www.kaggle.com/kshitijpatil09/pytorch-mean-absolute-precision-calculation?scriptVersionId=40816383


class MeanAveragePrecisionCalculator(object):
    def __init__(
        self,
        bounding_boxes_ground_truth,
        bounding_boxes_prediction,
        scores,
        thresholds: Set[torch.FloatTensor],
        value_below_threshold: torch.FloatTensor,
    ):
        self.bounding_boxes_ground_truth = bounding_boxes_ground_truth

        scores_order = scores.argsort().flip(-1)
        self.scores = scores[scores_order]
        self.bounding_boxes_prediction = bounding_boxes_prediction[scores_order]
        self.thresholds = thresholds

        log.debug(
            f"shape ground-truth {self.bounding_boxes_ground_truth.shape}, "
            f"prediction {self.bounding_boxes_prediction.shape}, "
            f"scores {self.scores.shape}"
        )

        self.iou_mat = MeanAveragePrecisionCalculator.calculate_iou(
            self.bounding_boxes_ground_truth, self.bounding_boxes_prediction
        )
        log.debug(f"iou_mat {self.iou_mat}")

        self.result = {}
        for threshold in thresholds:
            iou_mat = self.iou_mat.where(self.iou_mat > threshold, value_below_threshold)
            mappings = MeanAveragePrecisionCalculator.get_mappings(iou_mat)
            self.result[str(threshold.cpu().data.numpy())] = (
                MeanAveragePrecisionCalculator.calc_map(mappings).cpu().data.numpy().tolist()
            )

    @staticmethod
    def align_coordinates(boxes):
        """Align coordinates (x1,y1) < (x2,y2) to work with torchvision `box_iou` op
        Arguments:
            boxes (Tensor[N,4])

        Returns:
            boxes (Tensor[N,4]): aligned box coordinates
        """
        x1y1 = torch.min(boxes[:, :2,], boxes[:, 2:])
        x2y2 = torch.max(boxes[:, :2,], boxes[:, 2:])
        boxes = torch.cat([x1y1, x2y2], dim=1)
        return boxes

    @staticmethod
    def calculate_iou(gt, pr, form="pascal_voc"):
        if form == "coco":
            gt = gt.clone()
            pr = pr.clone()

            gt[:, 2] = gt[:, 0] + gt[:, 2]
            gt[:, 3] = gt[:, 1] + gt[:, 3]
            pr[:, 2] = pr[:, 0] + pr[:, 2]
            pr[:, 3] = pr[:, 1] + pr[:, 3]

        gt = MeanAveragePrecisionCalculator.align_coordinates(gt)
        pr = MeanAveragePrecisionCalculator.align_coordinates(pr)

        return box_iou(gt, pr)

    @staticmethod
    def get_mappings(iou_mat):
        gt_count, pr_count = iou_mat.shape

        mappings = torch.zeros_like(iou_mat)
        # first mapping (max iou for first pred_box)

        if pr_count > 0 and not iou_mat[:, 0].eq(0.0).all():
            # if not a zero column
            mappings[iou_mat[:, 0].argsort()[-1], 0] = 1

        for pr_idx in range(1, pr_count):
            # Sum of all the previous mapping columns will let
            # us know which gt-boxes are already assigned
            not_assigned = torch.logical_not(mappings[:, :pr_idx].sum(1)).long()

            # Considering unassigned gt-boxes for further evaluation
            targets = not_assigned * iou_mat[:, pr_idx]

            # If no gt-box satisfy the previous conditions
            # for the current pred-box, ignore it (False Positive)
            if targets.eq(0).all():
                continue

            # max-iou from current column after all the filtering
            # will be the pivot element for mapping
            pivot = targets.argsort()[-1]
            mappings[pivot, pr_idx] = 1
        return mappings

    @staticmethod
    def calc_map(mappings):
        tp = mappings.sum()
        fp = mappings.sum(0).eq(0).sum()
        fn = mappings.sum(1).eq(0).sum()
        mAP = tp / (tp + fp + fn)

        return mAP


# from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py


class EarlyStopping(object):
    def __init__(self, patience: int = 8, delta: float = 0.0, output_path: str = None):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.output_path = output_path
        self.epoch = None

    def __call__(self, val, model):
        # score = -val
        score = val  # mean average precision

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            log.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation mAP increases."""
        log.info(f"Validation mAP increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), path.join(self.output_path, "model_checkpoint.pt"))
        self.val_loss_min = val_loss
