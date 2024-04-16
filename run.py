from tqdm import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from utils import (combine_predictions,
                   nms, plot_image,
                   mean_average_precision)


def run(model, dataloader, loss_function, optimizer=None, device='cpu'):
    loop = tqdm(dataloader, leave=True)
    mean_loss = []
    all_pred_boxes = []
    all_true_boxes = []
    confidence_threshold = 0.5
    iou_threshold = 0.5
    train_idx = 0
    mode = 'midpoint'

    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_loss = 0

    for batch, (x, y) in enumerate(dataloader):
        batch_size = x.size()[0]
        x, y = x.to(device), y.to(device)
        pred = model(x)

        # loop for drawing bboxes on images
        true_boxes = combine_predictions(y)
        pred_boxes = combine_predictions(pred)
        for idx in range(batch_size):

            nms_boxes = nms(pred_boxes[idx], iou_threshold, confidence_threshold, mode)
            temp = []

            for nms_box in nms_boxes:
                # boxes for a given batch and image with index = idx
                all_pred_boxes.append([train_idx] + nms_box)

            # print(nms_boxes)
            plot_image(x[idx].permute(1, 2, 0).to("cpu"), nms_boxes)

            for box in true_boxes[idx]:
                if box[0] == 1:
                    all_true_boxes.append([train_idx] + box)
                    temp.append(box)

            # print(temp)
            plot_image(x[idx].permute(1, 2, 0).to("cpu"), temp)

            train_idx += 1

        import time
        time.sleep(10)
        clear_output()

        loss = loss_function(pred, y)
        mean_loss.append(loss.item())
        total_loss += loss.item()
        if optimizer is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        loop.set_postfix(loss=loss.item())

    mean_aver_precision = mean_average_precision(all_pred_boxes, all_true_boxes, iou_threshold=0.5,
                                                 box_format='midpoint')
    return total_loss / len(mean_loss), mean_aver_precision
