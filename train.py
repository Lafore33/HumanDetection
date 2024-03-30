import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as ft
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import YOLO
from dataset import ImageDataset
from iou import iou
from loss import Loss
from run import run, show_losses


seed = 123
torch.manual_seed(seed)

LR = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 100
train_images_path = "/Users/mikhailkoutun/Downloads/archive/images/train"
train_labels_path = "/Users/mikhailkoutun/Downloads/archive/labels/train"
val_images_path = "/Users/mikhailkoutun/Downloads/archive/images/val"
val_labels_path = "/Users/mikhailkoutun/Downloads/archive/labels/val"

transforms = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                 transforms.Resize((448, 448)),
                                 transforms.ToTensor()])


def main():
    model = YOLO().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_function = torch.nn.MSELoss(reduction='sum')

    train_dataset = ImageDataset(train_images_path, train_labels_path, transform=transforms)
    val_dataset = ImageDataset(val_images_path, val_labels_path, transform=transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    train_loss_hist = []

    for epoch in range(NUM_EPOCHS):
        mean_loss = run(model, train_loader, loss_function, optimizer, DEVICE)
        train_loss_hist.append(mean_loss)
        show_losses(train_loss_hist)


if __name__ == "__main__":
    main()
