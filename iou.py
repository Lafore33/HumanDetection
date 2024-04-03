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


def mean_average_precision(pred_boxes, true_boxes, iou_threshold, num_classes=1, box_format="midpoint"):
    average_precisions = []
    epsilon = 1e-6

    for i in range(num_classes):
        detections = []
        ground_truths = []
        amount_boxes = {}

        # here we assume pred_boxes as well as true_boxes have index at pos. 0
        for detection in pred_boxes:
            if detection[1] == i:
                detections.append(detection)

        for true_bbox in true_boxes:
            if true_bbox[1] == i:
                ground_truths.append(true_bbox)
                amount_boxes[true_bbox[0]] = amount_boxes.get(true_bbox[0], 0) + 1

        amount_boxes = {key: torch.zeros(val) for key, val in amount_boxes.items()}
        detections.sort(key=lambda x: x[2], reverse=True)
        true_positives = torch.zeros(len(detections))
        false_positives = torch.zeros(len(detections))
        total_true_boxes = len(ground_truths)

        for idx, detection in enumerate(detections):
            gt_img = [gt for gt in ground_truths if gt[0] == detection[0]]

            best_iou = 0
            best_gt_idx = 0

            for gt_idx, gt in enumerate(gt_img):
                cur_iou = iou(torch.tensor(detection[3:]), torch.tensor(gt[3:]), box_format)

                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_gt_idx = gt_idx

            if best_iou > iou_threshold:
                if amount_boxes[detection[0]][best_gt_idx] == 0:
                    true_positives[detection[0]] = 1
                    amount_boxes[detection[0]][best_gt_idx] = 1
            else:
                false_positives[idx] = 1

        cum_tp = torch.cumsum(true_positives, dim=0)
        cum_fp = torch.cumsum(false_positives, dim=0)

        recall = cum_tp / (total_true_boxes + epsilon)
        precision = cum_tp / (cum_fp + cum_tp + epsilon)
        precisions = torch.cat((torch.tensor([1]), precision))
        recalls = torch.cat((torch.tensor([0]), recall))
        average_precisions.append(torch.trapz(precisions, recalls))
        return sum(average_precisions) / len(average_precisions)

