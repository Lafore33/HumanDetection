from tqdm import tqdm
import torch
from utils import combine_pred, non_max_suppression


def train_fn(model, loader, loss_fn, optimizer=None, device='cpu'):
    loop = tqdm(loader, leave=True)
    mean_loss = []

    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_loss = 0

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)

        if optimizer is not None:
            optimizer.zero_grad()

        raw_pred = model(x)
        print(raw_pred.shape)
        loss = loss_fn(raw_pred, y)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        mean_loss.append(loss.item())
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(mean_loss)


def get_bboxes(model, loader, iou_threshold, class_threshold, device='cpu'):
    all_true_boxes = []
    all_pred_boxes = []
    train_idx = 0

    model.eval()

    for batch_idx, (x, y) in enumerate(loader):
        batch_size = x.size()[0]
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            raw_pred = model(x)
            pred = combine_pred(raw_pred)
            y = combine_pred(y)
            for idx in range(batch_size):
                nms_boxes = non_max_suppression(pred[idx], iou_threshold, class_threshold)
                temp = []

                for nms_box in nms_boxes:
                    # boxes for a given batch and image with index = idx
                    all_pred_boxes.append([train_idx] + nms_box)

                # plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)

                for box in y[idx]:
                    if box[0] == 1:
                        all_true_boxes.append([train_idx] + box)
                        temp.append(box)

                # plot_image(x[idx].permute(1,2,0).to("cpu"), temp)

                train_idx += 1

    return all_pred_boxes, all_true_boxes
