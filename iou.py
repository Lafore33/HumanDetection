# file for intersection_over_union
import torch


def calculate_coordinates(boxes, mode="midpoint"):
    x1, y1, x2, y2 = 0, 0, 0, 0

    if mode == "midpoint":
        x1 = boxes[..., 0:1] - boxes[..., 2:3] / 2
        y1 = boxes[..., 1:2] - boxes[..., 3:4] / 2
        x2 = boxes[..., 0:1] + boxes[..., 2:3] / 2
        y2 = boxes[..., 1:2] + boxes[..., 3:4] / 2

    elif mode == "corners":
        x1 = boxes[..., 0:1]
        y1 = boxes[..., 1:2]
        x2 = boxes[..., 2:3]
        y2 = boxes[..., 3:4]

    return x1, y1, x2, y2


def iou(boxes_pred, true_boxes, mode="midpoint"):
    epsilon = 1e-6
    pred_box_x1, pred_box_y1, pred_box_x2, pred_box_y2 = 0, 0, 0, 0
    true_box_x1, true_box_y1, true_box_x2, true_box_y2 = 0, 0, 0, 0
    if mode == "midpoint":
        pred_box_x1, pred_box_y1, pred_box_x2, pred_box_y2 = calculate_coordinates(boxes_pred)
        true_box_x1, true_box_y1, true_box_x2, true_box_y2 = calculate_coordinates(true_boxes)

    elif mode == "corners":
        pred_box_x1, pred_box_y1, pred_box_x2, pred_box_y2 = calculate_coordinates(boxes_pred, mode="corners")
        true_box_x1, true_box_y1, true_box_x2, true_box_y2 = calculate_coordinates(true_boxes, mode="corners")

    x1 = torch.max(pred_box_x1, true_box_x1)
    y1 = torch.max(pred_box_y1, true_box_y1)
    x2 = torch.min(pred_box_x2, true_box_x2)
    y2 = torch.min(pred_box_y2, true_box_y2)

    # what is .clamp(0) for?
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((pred_box_x2 - pred_box_x1) * (pred_box_y2 - pred_box_y1))
    box2_area = abs((true_box_x2 - true_box_x1) * (true_box_y2 - true_box_y1))

    return intersection / (box1_area + box2_area - intersection + epsilon)


nums1 = torch.randn(1, 7, 7, 4)
nums2 = torch.randn(1, 7, 7, 4)
nums3 = torch.randn(1, 7, 7, 4)
nums4 = torch.randn(1, 7, 7, 4)
iou1 = iou(nums1, nums2)
iou2 = iou(nums3, nums4)
print(torch.cat((iou1.unsqueeze(0), iou2.unsqueeze(0)), dim=0))
