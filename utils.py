import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch


def convert_to_corners(boxes):
    """ Converts boxes with midpoint format to corner format"""

    x1 = boxes[..., 0:1] - boxes[..., 2:3] / 2
    y1 = boxes[..., 1:2] - boxes[..., 3:4] / 2
    x2 = boxes[..., 0:1] + boxes[..., 2:3] / 2
    y2 = boxes[..., 1:2] + boxes[..., 3:4] / 2

    return x1, y1, x2, y2


def intersection_over_union(boxes_predictions, boxes_labels):
    """
    Calculates intersection over union

    Parameters:
        boxes_predictions (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
    Returns:
        tensor: Intersection over union for all examples
    """
    box1_x1, box1_y1, box1_x2, box1_y2 = convert_to_corners(boxes_predictions)
    box2_x1, box2_y1, box2_x2, box2_y2 = convert_to_corners(boxes_labels)

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, x, y, w, h]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert isinstance(bboxes, list)

    bboxes = [box for box in bboxes if box[0] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[0], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if intersection_over_union(
                torch.tensor(chosen_box[1:]),
                torch.tensor(box[1:])) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

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


# функции вывода графиков
def show_losses(train_loss_hist, test_loss_hist=None):
    #     clear_output()

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


def convert_bboxes(pred, s=7):

    pred = pred.to('cpu')
    cell_indices = torch.arange(7).repeat(1, 7, 1).unsqueeze(-1)

    pred[..., 1:2] = 1 / s * (pred[..., 1:2] + cell_indices)
    pred[..., 2:3] = 1 / s * (pred[..., 2:3] + cell_indices.permute(0, 2, 1, 3))
    pred[..., 3:5] = 1 / s * (pred[..., 3:5])

    return pred


def combine_pred(out, s=7):

    batch_size = out.shape[0]
    pred = convert_bboxes(out).reshape(batch_size, s * s, -1)
    all_bboxes = []

    for idx in range(batch_size):
        bboxes = []

        for bbox_idx in range(s * s):
            bboxes.append([x.item() for x in pred[idx, bbox_idx, :]])

        all_bboxes.append(bboxes)

    return all_bboxes
