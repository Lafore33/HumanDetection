import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from model import YOLO
from loss import YoloLoss
from dataset import ImageDataset
from train import train_fn, get_bboxes
from mean_average_precision import mean_average_precision


seed = 123
torch.manual_seed(seed)

LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 1000
iou_threshold = 0.4
class_threshold = 0.5

train_images_path = "detection/images/train"
train_labels_path = "detection/labels/train"
val_images_path = "detection/images/val"
val_labels_path = "detection/labels/val"

transforms = transforms.Compose([transforms.Resize((448, 448)),
                                 transforms.ToTensor()])

model = YOLO().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_function = YoloLoss()
# loss_function = Loss()

train_dataset = ImageDataset(train_images_path, train_labels_path, transform=transforms)
val_dataset = ImageDataset(val_images_path, val_labels_path, transform=transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

train_loss_hist = []


def main():
    for epoch in range(NUM_EPOCHS):
        mean_loss = train_fn(model, train_loader, loss_function, optimizer, DEVICE)

        all_pred_boxes, all_true_boxes = get_bboxes(model, train_loader, iou_threshold, class_threshold, DEVICE)

        mean_avg_precision = mean_average_precision(all_pred_boxes, all_true_boxes, iou_threshold)
        #     train_loss_hist.append(mean_loss)
        #     show_losses(train_loss_hist)
        print(f'Epoch {epoch}; Mean average precision: {round(mean_avg_precision, 2)}')


if __name__ == "__main__":
    main()


