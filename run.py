from tqdm import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import torch
from iou import iou


def run(model, dataloader, loss_function, optimizer=None, device='cpu'):

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
        best_boxes = torch.where(pred[..., 1:2] > pred[..., 6:7], pred[..., 1:6], pred[..., 6:11])
        predicted_class = torch.where(pred[..., 0:1] > 0.5, 1, 0)
        converted_pred = torch.cat((predicted_class, best_boxes), dim=-1)
        loss = loss_function(torch.flatten(converted_pred), torch.flatten(y[..., 0:6]))
        mean_loss.append(loss.item())
        total_loss += loss.item()
        if optimizer is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        loop.set_postfix(loss=loss.item())

    return total_loss / len(mean_loss)


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
    best_boxes = torch.where(pred[..., 1:2] > pred[..., 6:7], pred[..., 1:6], pred[..., 6:11])
    cell_indices = torch.arange(7).repeat(1, 7, 1).unsqueeze(-1)
    # x
    best_boxes[..., 1:2] = 1/s * (best_boxes[..., 1:2] + cell_indices)
    # y
    best_boxes[..., 2:3] = 1/s * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    # height and width
    best_boxes[..., 3:5] = 1/s * best_boxes[..., 3:5]
    predicted_class = torch.where(pred[..., 0:1] > 0.5, 1, 0)
    converted_pred = torch.cat((predicted_class, best_boxes), dim=-1)
    return converted_pred


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

    boxes = [box for box in pred_boxes if box[1] > confidence_threshold]
    boxes.sort(key=lambda x: x[1], reverse=True)
    res = []

    while boxes:

        cur_best_box = boxes.pop(0)
        boxes = [box for box in boxes if cur_best_box[0] != box[0] or
                 iou(torch.tensor(cur_best_box[2:]), torch.tensor(box[2:]), mode) < iou_threshold]
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

        batch_size = x.size(0)
        true_boxes = combine_predictions(y)
        pred_boxes = combine_predictions(pred)

        for idx in range(batch_size):

            nms_boxes = nms(pred_boxes[idx], iou_threshold, confidence_threshold, mode)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_boxes[idx]:
                if box[1] > confidence_threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1
    # ??
    model.train()
    return all_pred_boxes, all_true_boxes
