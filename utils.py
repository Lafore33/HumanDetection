# file for intersection_over_union
import torch
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        box = box[1:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()


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

    detections = []
    ground_truths = []
    amount_boxes = {}
    # here we assume pred_boxes as well as true_boxes have index at pos. 0
    for detection in pred_boxes:
        if detection[1] > 0.5:
            detections.append(detection)
    for true_bbox in true_boxes:
        ground_truths.append(true_bbox)
        amount_boxes[true_bbox[0]] = amount_boxes.get(true_bbox[0], 0) + 1
    amount_boxes = {key: torch.zeros(val) for key, val in amount_boxes.items()}
    detections.sort(key=lambda x: x[1], reverse=True)
    true_positives = torch.zeros(len(detections))
    false_positives = torch.zeros(len(detections))
    total_true_boxes = len(ground_truths)
    for idx, detection in enumerate(detections):
        gt_img = [gt for gt in ground_truths if gt[0] == detection[0]]
        best_iou = 0
        best_gt_idx = 0
        for gt_idx, gt in enumerate(gt_img):
            cur_iou = iou(torch.tensor(detection[2:]), torch.tensor(gt[2:]), box_format)
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


# функции вывода графиков
def show_losses(train_loss_hist, test_loss_hist=None):
    clear_output()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.title('Train Loss')
    plt.plot(np.arange(len(train_loss_hist)), train_loss_hist)
    plt.yscale('log')
    plt.grid()

    if test_loss_hist is not None:
        plt.subplot(1, 2, 2)
        plt.title('Test Loss')
        plt.plot(np.arange(len(test_loss_hist)), test_loss_hist)
        plt.yscale('log')
        plt.grid()

    plt.show()


# function for converting coordinates of the bboxes relative to cells to relative to the image itself
def convert_coordinates(pred, s=7):
    pred = pred.to('cpu')
    batch_size = pred.shape[0]
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    # x
    pred[..., 1:2] = 1 / s * (pred[..., 1:2] + cell_indices)
    # y
    pred[..., 2:3] = 1 / s * (pred[..., 2:3] + cell_indices.permute(0, 2, 1, 3))
    # height and width
    pred[..., 3:5] = 1 / s * (pred[..., 3:5])
    return pred


# function for combining all bounding boxes for one image into a 49x6 matrix, so we have a matrix of all bboxes
def combine_predictions(pred, s=7):
    converted_pred = convert_coordinates(pred).reshape(pred.shape[0], s * s, -1)
    res = []
    for i in range(converted_pred.shape[0]):
        boxes = []

        for box_index in range(s * s):
            boxes.append([box.item() for box in converted_pred[i, box_index, :]])

        res.append(boxes)
    return res


def nms(pred_boxes, iou_threshold, confidence_threshold, mode='midpoint'):
    boxes = [box for box in pred_boxes if box[0] > confidence_threshold]
    boxes.sort(key=lambda x: x[0], reverse=True)
    res = []

    while boxes:
        cur_best_box = boxes.pop(0)
        boxes = [box for box in boxes if
                 iou(torch.tensor(cur_best_box[1:]), torch.tensor(box[1:]), mode) < iou_threshold]
        res.append(cur_best_box)

    return res


def get_boxes(loader, model, iou_threshold, confidence_threshold, device='cpu', mode='midpoint', train=True):
    # if not train:
    #    model.eval()
    # else:
    #    model.train()
    # ??
    model.eval()

    all_true_boxes = []
    all_pred_boxes = []

    train_idx = 0

    for batch, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            pred = model(x)

        batch_size = x.size[0]
        true_boxes = combine_predictions(y)
        pred_boxes = combine_predictions(pred)

        for idx in range(batch_size):

            nms_boxes = nms(pred_boxes[idx], iou_threshold, confidence_threshold, mode)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_boxes[idx]:
                if box[0] > confidence_threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1
    # ??
    model.train()
    return all_pred_boxes, all_true_boxes
