# file for the loss function

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
        self.cross = nn.CrossEntropyLoss(reduction='sum')
        self.cells = cells
        self.boxes = boxes
        self.classes = classes
        self.lambda_no_obj = 0.5
        self.lambda_coord = 5

    def forward(self, pred_boxes, true_boxes):
        return (self.mse(pred_boxes[self.classes:], true_boxes[self.classes:]) +
                self.lambda_coord * self.cross(pred_boxes[:self.classes], true_boxes[:self.classes]))

