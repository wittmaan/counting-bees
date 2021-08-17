import logging.config
from collections import defaultdict

import numpy as np
import torch
import torchvision
from torch import tensor

from common.util import AverageMeter, MeanAveragePrecisionCalculator

logging.config.fileConfig(fname="../resources/logging.conf", disable_existing_loggers=False)
log = logging.getLogger("engine")

MB = 1024.0 * 1024.0


def train_one_epoch(
    model, optimizer, data_loader: torch.utils.data.DataLoader, device, epoch, epochs, lr_scheduler, print_freq
):
    model.detector()
    loss_average = AverageMeter()

    for i, (images, targets, file_names) in enumerate(data_loader):
        log.debug(f"file_names {file_names}")

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_average.update(losses.item(), len(images))

        # clear gradients of all optimized variables
        optimizer.zero_grad()
        # backward pass: compute gradient of the loss with respect to the model parameters
        losses.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # decay learning rate
        lr_scheduler.step()

        if i % print_freq == 0:
            log.info(
                f"Epoch: [{epoch + 1}/{epochs}]\t"
                f"Batch [{i}/{len(data_loader)}]\t "
                f"Loss {loss_average.avg:.4f}\t "
                f"Learning-Rate {optimizer.param_groups[0]['lr']:.4f}\t"
                f"Memory allocated {torch.cuda.max_memory_allocated() / MB:.0f}"
            )


@torch.no_grad()
def evaluate(
    model: torchvision.models.detection.fasterrcnn_resnet50_fpn,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
):
    model.eval()
    thresholds = (tensor(0.5).to(device), tensor(0.75).to(device))

    map_result = defaultdict(list)
    for i, (images, targets, file_names) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        predictions = model(images)
        log.debug(f"predictions {predictions}")

        for idx, prediction in enumerate(predictions):
            bounding_boxes_ground_truth = targets[0]["boxes"]  # .cpu().data.numpy()
            bounding_boxes_prediction = prediction["boxes"]  # .cpu().data.numpy()
            scores = prediction["scores"]  # .cpu().data.numpy()

            calc = MeanAveragePrecisionCalculator(
                bounding_boxes_ground_truth=bounding_boxes_ground_truth,
                bounding_boxes_prediction=bounding_boxes_prediction,
                scores=scores,
                thresholds=thresholds,
                value_below_threshold=tensor(0.0).to(device),
            )
            # log.info(f"mAP {calc.mAP}")
            # log.info(f"mAP {calc.result}")
            for k in calc.result:
                map_result[k].append(calc.result[k])
    # log.info(f"map_result {map_result}")
    map_mean_result = defaultdict(np.ndarray)
    for k in map_result:
        map_mean_result[k] = np.mean(map_result[k])
    log.info(f"map_mean_result {map_mean_result}")

    return map_mean_result
