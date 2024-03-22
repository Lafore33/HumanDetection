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

        # box_loss
        # pred_boxes[..., 2:6].shape = (N, 7, 7, 4) here we extract x, y, w, h of the box
        iou1 = iou(pred_boxes[..., 2:6], true_boxes[..., 2:6])
        iou2 = iou(pred_boxes[..., 7:11], true_boxes[..., 2:6])

        best_iou, best_box = torch.max(torch.cat((iou1, iou2), dim=0), dim=0)
        exist_box = true_boxes[..., 1]
        box_pred = exist_box * ((1 - best_box) * pred_boxes[2:6] + best_box * pred_boxes[7:11])
        box_true = exist_box * true_boxes[2:6]
        box_pred[..., 2:4] = torch.sign(box_pred[..., 2:4]) * torch.sqrt(torch.abs(box_pred[..., 2:4]) + epsilon)
        box_true[..., 2:4] = torch.sqrt(box_true[..., 2:4])
        box_loss = self.mse(box_pred, box_true)

        # confidence loss
        confidence_box_pred = (1 - best_box) * pred_boxes[..., 1:2] + best_box * pred_boxes[..., 6:7]
        confidence_loss = self.mse(exist_box * confidence_box_pred, exist_box * true_boxes[..., 1:2])

        # no object loss
        no_obj_loss = self.mse((1 - exist_box) * box_pred[..., 1:2], (1 - exist_box) * true_boxes[..., 1:2])
        no_obj_loss += self.mse((1 - exist_box) * box_pred[..., 6:7], (1 - exist_box) * true_boxes[..., 1:2])

        # class loss
        class_loss = self.mse(exist_box * box_pred[..., :1], exist_box * true_boxes[..., :1])

        total_loss = box_loss * self.lambda_coord + confidence_loss + self.lambda_no_obj * no_obj_loss + class_loss
        return total_loss
