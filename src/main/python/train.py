import argparse
import logging.config
from datetime import timedelta
from os import path
from time import time

import pandas as pd
import torch
import torchvision
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.sgd import SGD
from torch.utils.data.dataloader import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from common.util import collate_fn, EarlyStopping, worker_init_fn
from dataset import BeeDataset
from engine import train_one_epoch, evaluate

logging.config.fileConfig(fname="../resources/logging.conf", disable_existing_loggers=False)
log = logging.getLogger("train")


class BeeTrainingApp(object):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cuda_is_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_is_available else "cpu")
        log.info(f"cuda_is_available={self.cuda_is_available}, device={self.device}")

        self.model = self.init_model()
        self.optimizer = self.init_optimizer()

    def init_model(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, box_detections_per_img=1000)
        num_classes = 2  # 1 class (bee) + background
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        if self.cuda_is_available:
            log.info("found {} device(s) for cuda".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)

            model = model.to(self.device)

        log.info("model initialized")
        return model

    def init_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005, nesterov=True)

        log.info("optimizer initialized")
        return optimizer

    def init_data_loader(self, data_type: str):
        ds = None
        if data_type == "train":
            df = pd.read_csv(path.join(self.args.input_path, "annotations_train.csv"))
            ds = BeeDataset(df, path.join(self.args.input_path, "images"), do_augmentation=True)
        elif data_type == "valid":
            df = pd.read_csv(path.join(self.args.input_path, "annotations_valid.csv"))
            ds = BeeDataset(df, path.join(self.args.input_path, "images"), do_augmentation=False)

        data_loader = DataLoader(
            dataset=ds,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.cuda_is_available,
            shuffle=True,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
        )
        log.info(f"initialized data_loader for {data_type}")
        return data_loader

    def run(self):
        train_data_loader = self.init_data_loader(data_type="train")
        valid_data_loader = self.init_data_loader(data_type="valid")
        lr_scheduler = MultiStepLR(self.optimizer, milestones=self.args.lr_steps, gamma=self.args.lr_gamma)
        early_stopping = EarlyStopping(patience=50, output_path=self.args.output_dir)

        start_time = time()
        for epoch in range(self.args.epochs):
            train_one_epoch(
                self.model,
                self.optimizer,
                train_data_loader,
                self.device,
                epoch,
                self.args.epochs,
                lr_scheduler,
                self.args.print_freq,
            )

            map_mean_result = evaluate(self.model, valid_data_loader, self.device)

            early_stopping(map_mean_result["0.5"], self.model)

            if early_stopping.early_stop:
                log.info("Early stopping")
                break

        total_time = time() - start_time
        total_time_str = str(timedelta(seconds=int(total_time)))
        log.info(f"Training time {total_time_str}")


if __name__ == "__main__":
    log.info("Training - start")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="path of the input data", default="../../../data/raw/")
    parser.add_argument(
        "--num-workers", help="Number of worker processes for background data loading", default=4, type=int,
    )
    parser.add_argument(
        "--batch-size", help="Batch size to use for training", default=1, type=int,
    )
    parser.add_argument(
        "--epochs", help="Number of epochs to train for", default=100000, type=int,
    )
    parser.add_argument(
        "--lr-steps", default=[100000, 200000], nargs="+", type=int, help="decrease lr every step-size epochs"
    )
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="../../../data/training", help="path where to save")
    parsed_args = parser.parse_args()
    log.info("got the following args: {}".format(parsed_args))

    train = BeeTrainingApp(parsed_args)
    train.run()

    log.info("Traning - end")
