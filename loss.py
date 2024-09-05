import torch
from torch import nn


class YoloLoss(nn.Module):

    def __init__(self):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.lambda_no_obj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        exists_box = target[..., 0:1]

        box_predictions = exists_box * predictions[..., 1:5]
        box_targets = exists_box * target[..., 1:5]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(box_predictions, box_targets)

        temp_pred_obj = exists_box * predictions[..., 0:1]
        temp_target_obj = exists_box * target[..., 0:1]
        class_loss = self.mse(temp_pred_obj, temp_target_obj)

        temp_pred = (1 - exists_box) * predictions[..., 0:1]
        temp_target = (1 - exists_box) * target[..., 0:1]

        no_object_class_loss = self.mse(temp_pred, temp_target)

        loss = self.lambda_coord * box_loss + class_loss + self.lambda_no_obj * no_object_class_loss

        return loss
