import torch
from torch import nn


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.cross = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, predictions, target):
        exist_bbox = target[..., 0:1]
        object_loss = self.mse(exist_bbox * predictions[..., 1:], target[..., 1:])

        # no_object_loss = self.mse((1 - exist_bbox) * predictions[..., 0:1], target[..., 0:1])
        class_loss = self.mse(predictions[..., 0:1], target[..., 0:1])

        # no_object_loss = self.mse((1 - exist_bbox) * predictions[..., 1:], target[..., 1:])

        return 5.0 * object_loss + class_loss


class YoloLoss(nn.Module):

    def __init__(self):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.cross = nn.CrossEntropyLoss(reduction="sum")
        self.lambda_no_obj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        exists_box = target[..., 0:1]

        box_predictions = exists_box * predictions[..., 1:5]
        box_targets = exists_box * target[..., 1:5]

        box_predictions[..., 2:4] = (torch.sign(box_predictions[..., 2:4]) *
                                     torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6)))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(box_predictions, box_targets)
        class_loss = self.cross((exists_box * predictions[..., 0:1]), (exists_box * target[..., 0:1]))

        #         no_object_loss = self.mse(((1 - exists_box) * predictions[..., 0:1]),
        #                                   ((1 - exists_box) * target[..., 0:1]))

        loss = self.lambda_coord * box_loss + class_loss

        return loss

