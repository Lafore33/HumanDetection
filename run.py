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
        loss = loss_function(pred, y)
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


# function for converting coordinates of the bboxes relative to cells to relative to the image itself
def convert_coordinates(predictions, s=7):
    predictions = predictions.to('cpu')
    batch = predictions.shape[0]
    pred = torch.tensor(predictions.reshape(batch, s, s, 11))
    best_boxes = torch.where(pred[..., 1:2] > pred[..., 6:7], pred[..., 1:6], pred[..., 6:11])
    cell_indices = torch.arange(7).repeat(1, 7, 1).unsqueeze(-1)
    # x
    best_boxes[..., 1:2] = 1/s * (best_boxes[..., 1:2] + cell_indices)
    # y
    best_boxes[..., 2:3] = 1/s * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    # height and width
    best_boxes[..., 3:5] = 1/s * best_boxes[..., 3:5]
    predicted_class = torch.where(predictions[0] > 0.5, 1, 0)
    converted_pred = torch.cat((predicted_class, best_boxes), dim=-1)
    return converted_pred


# function for combining all bounding boxes for one image into a 49x11 matrix, so we have a matrix of all bboxes
def combine_predictions(pred, s=7):
    combined_pred = pred.reshape(pred.shape[0], s * s, -1)
    res = []
    for i in range(combined_pred.shape[0]):
        boxes = []

        for box_index in range(s * s):
            boxes.append([box for box in combined_pred[i, box_index, :]])

        res.append(boxes)
    return res


