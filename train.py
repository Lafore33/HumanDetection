import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as ft
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import YOLO
from dataset import ImageDataset
from run import run
from utils import show_losses

seed = 123
torch.manual_seed(seed)

LR = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 100


train_images_path = "/Users/Downloads/HumanDataset/train/images"
train_labels_path = "/Users/Downloads/HumanDataset/train/labels"
val_images_path = "/Users/Downloads/HumanDataset/val/images"
val_labels_path = "/Users/Downloads/HumanDataset/val/labels"


transforms = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                 transforms.Resize((448, 448)),
                                 transforms.ToTensor()])


def main():
    model = YOLO().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_function = torch.nn.MSELoss(reduction='sum')
    # loss_function = Loss()

    train_dataset = ImageDataset(train_images_path, train_labels_path, transform=transforms)
    val_dataset = ImageDataset(val_images_path, val_labels_path, transform=transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    train_loss_hist = []
    val_loss_hist = []
    mean_aver_precision = 0

    for epoch in range(NUM_EPOCHS):
        print(mean_aver_precision)
        import time
        # time.sleep(10)
        mean_loss, mean_aver_precision = run(model, train_loader, loss_function, optimizer, DEVICE)
        train_loss_hist.append(mean_loss)
        val_mean_loss, val_mean_aver_precision = run(model, val_loader, loss_function, None, DEVICE)
        val_loss_hist.append(val_mean_loss)
        show_losses(train_loss_hist, val_loss_hist)
        path = f'./model{epoch}.pt'
        torch.save(model.state_dict(), path)
        time.sleep(20)


if __name__ == "__main__":
    main()
