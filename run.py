from tqdm import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import torch
from iou import iou


def run(model, dataloader, loss, optimizer=None, device='cpu'):

    loop = tqdm(dataloader, leave=True)
    mean_loss = []

    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_loss = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        current_loss = loss(pred, y)
        mean_loss.append(current_loss.item())
        total_loss += current_loss.item()
        if optimizer is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        loop.set_postfix(loss=current_loss.item())

    return total_loss / len(mean_loss)


# функции вывода графиков
def show_losses(train_loss_hist, test_loss_hist=None):
    clear_output()

    plt.figure(figsize=(12,4))

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


def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=1):
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truth = []
        total_boxes = {}
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                total_boxes[true_box[0]] = total_boxes.get(true_box[0], 0) + 1
                ground_truth.append(true_box)

        total_boxes = {key: torch.zeros(total_boxes[key]) for key in total_boxes}

        detections.sort(key=lambda x: x[2], reverse=True)

        true_positives = torch.zeros(len(detections))
        false_positives = torch.zeros(len(detections))

        total_true_boxes = len(ground_truth)

        if not total_true_boxes:
            continue

        for idx, detection in enumerate(detections):

            ground_truth_img = [bbox for bbox in ground_truth if bbox[0] == detection[0]]

            best_iou = 0
            best_gt_index = -1
            for gt_idx, gt_box in enumerate(ground_truth_img):

                cur_iou = iou(torch.tensor(detection[3:]), torch.tensor(gt_box[3:]), box_format)
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_gt_index = gt_idx

            if best_iou > iou_threshold:
                if total_boxes[detection[0]][best_gt_index] == 0 and best_gt_index != -1:
                    total_boxes[detection[0]][best_gt_index] = 1
                    true_positives[idx] = 1
            else:
                false_positives[idx] = 1

        tp_sum = torch.cumsum(true_positives, dim=0)
        fp_sum = torch.cumsum(false_positives, dim=0)
        recalls = tp_sum / (total_true_boxes + epsilon)
        precisions = tp_sum / (tp_sum + fp_sum + epsilon)
        recalls = torch.cat((torch.tensor([0]), recalls))
        precisions = torch.cat((torch.tensor([1]), precisions))
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)




