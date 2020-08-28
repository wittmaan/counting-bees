import logging

import pytest
import torch
from torch import tensor
from torchvision.ops import box_iou

from common.util import MeanAveragePrecisionCalculator

log = logging.getLogger("test_utils")


class TestMeanAveragePrecisionCalculator(object):
    @pytest.fixture(autouse=True)
    def resource(self):
        log.info("setup")
        self.cuda_is_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_is_available else "cpu")

        yield
        log.info("teardown")

    def test_case1(self):
        predictions = (
            tensor(
                [
                    [260.0, 350.0, 290.0, 400.0],
                    [220.0, 300.0, 200.0, 390.0],
                    [120.0, 180.0, 160.0, 220.0],
                    [80.0, 110.0, 130.0000, 200.0],
                ]
            )
            .float()
            .to(self.device)
        )

        ground_truth = (
            tensor(
                [
                    [245.6250, 344.3182, 279.3750, 389.7727],
                    [218.7500, 332.9546, 241.8750, 384.0909],
                    [123.7500, 168.1818, 157.5000, 213.6364],
                    [83.7500, 150.0000, 120.0000, 197.7273],
                    [116.2500, 134.0909, 148.7500, 173.8636],
                    [81.2500, 131.8182, 115.0000, 164.7727],
                    [90.6250, 93.1818, 113.7500, 135.2273],
                    [111.2500, 106.8182, 141.8750, 134.0909],
                    [145.0000, 96.5909, 166.2500, 160.2273],
                    [114.3750, 75.0000, 148.1250, 120.4545],
                ]
            )
            .float()
            .to(self.device)
        )

        scores = tensor([0.90, 0.80, 0.90, 0.99,]).to(self.device)

        thresholds = (tensor(0.5).to(self.device), tensor(0.75).to(self.device))

        calc = MeanAveragePrecisionCalculator(
            bounding_boxes_ground_truth=ground_truth,
            bounding_boxes_prediction=predictions,
            scores=scores,
            thresholds=thresholds,
            value_below_threshold=tensor(0.0).to(self.device),
        )

        expected_result = {"0.5": 0.07692307978868484, "0.75": 0.0}
        assert calc.result == expected_result

    def test_case2(self):
        ground_truth = torch.rand(10, 4)
        prediction = torch.rand(2, 4)

        ground_truth = MeanAveragePrecisionCalculator.align_coordinates(ground_truth)
        prediction = MeanAveragePrecisionCalculator.align_coordinates(prediction)
        iou_mat = box_iou(ground_truth, prediction)

        mappings = MeanAveragePrecisionCalculator.get_mappings(iou_mat)
        mAP = MeanAveragePrecisionCalculator.calc_map(mappings)

        assert torch.equal(mAP, tensor(0.2))

    def test_case3(self):
        ground_truth = torch.rand(10, 4)
        prediction = torch.rand(0, 4)

        ground_truth = MeanAveragePrecisionCalculator.align_coordinates(ground_truth)
        prediction = MeanAveragePrecisionCalculator.align_coordinates(prediction)
        iou_mat = box_iou(ground_truth, prediction)

        mappings = MeanAveragePrecisionCalculator.get_mappings(iou_mat)
        mAP = MeanAveragePrecisionCalculator.calc_map(mappings)

        assert torch.equal(mAP, tensor(0.0))

    # def test_early_stopping(self):
    #     early_stopping = EarlyStopping()
