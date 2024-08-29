import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from model import YOLO, YoloResNet
from loss import YoloLoss, Loss
from dataset import ImageDataset
import cv2 as cv
from train import train_fn, get_bboxes
from mean_average_precision import mean_average_precision

LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_EPOCHS = 15


def main():
    seed = 123
    torch.manual_seed(seed)
    train_images_path = "detection/images/train"
    train_labels_path = "detection/labels/train"
    val_images_path = "detection/images/val"
    val_labels_path = "detection/labels/val"

    transform_aug = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ])
    model = YoloResNet().to(DEVICE)
    # model = YOLO().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.0005)
    loss_function = YoloLoss()
    # loss_function = Loss()

    train_dataset = ImageDataset(train_images_path, train_labels_path, transform=transform_aug)
    val_dataset = ImageDataset(val_images_path, val_labels_path, transform=transform_aug)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    iou_threshold = 0.4
    class_threshold = 0.5

    train_loss_hist = []
    for epoch in range(NUM_EPOCHS):
        mean_loss = train_fn(model, train_loader, loss_function, optimizer, DEVICE)

        all_pred_boxes, all_true_boxes = get_bboxes(model, train_loader, iou_threshold, class_threshold, DEVICE)

        mAp = mean_average_precision(all_pred_boxes, all_true_boxes, iou_threshold)

        #     train_loss_hist.append(mean_loss)
        #     show_losses(train_loss_hist)
        print(f'Epoch {epoch}; Mean average precision: {round(mAp, 2)}')



if __name__ == '__main__':
    main()
