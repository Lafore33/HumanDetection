# file for loss function for the model

import numpy as np
import torch
from iou import iou
from torch import nn
lambda_coord = 5
lambda_no_obj = 0.5


class Loss(nn.Module):
    def __init__(self, cells=7, boxes=2, classes=1):
        super(Loss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.cells = cells
        self.boxes = boxes
        self.classes = classes
        self.lambda_no_obj = 0.5
        self.lambda_coord = 5

    def forward(self, pred_boxes, true_boxes):
        epsilon = 1e-6
        # (N, 7, 7, 11)
        pred_boxes = pred_boxes.reshape(-1, self.cells, self.cells, self.boxes * 5 + self.classes)
        # iou's are going to be of the shape of (N, 7, 7, 1)
        iou1 = iou(pred_boxes[..., 2:6], true_boxes[..., 2:6])
        iou2 = iou(pred_boxes[..., 7:11], true_boxes[..., 2:6])


        pass