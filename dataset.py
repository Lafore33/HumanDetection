import numpy as np
import pandas as pd
import torch
import os
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from model import Model
from torchsummary import summary
from torch import nn
import cv2


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, image_root, label_root, transform=None):
        self.image_root = image_root
        self.label_root = label_root
        self.titles = [title for title in os.listdir(image_root)]
        self.transform = transform

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, index):
        image_file = self.titles[index]
        image_path = os.path.join(self.image_root, image_file)
        img = Image.open(image_path)
        if self.transform is not None:
            img = self.transform(img)

        label_file = self.titles[index][:-3] + "txt"
        label_path = os.path.join(self.label_root, label_file)

        with open(label_path, 'r') as file:
            label = file.read().strip()
            label = label.split("\n")
            label = [np.array(vector[2:].split(), dtype="float64") for vector in label]

        return img, label


train_images_path = "/Users/mikhailkoutun/Downloads/archive/images/train"
val_images_path = "/Users/mikhailkoutun/Downloads/archive/images/val"

train_labels_path = "/Users/mikhailkoutun/Downloads/archive/labels/train"
val_labels_path = "/Users/mikhailkoutun/Downloads/archive/labels/val"
batch_size = 64

transforms = transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=3),
                                 torchvision.transforms.Resize((448, 448)),
                                 torchvision.transforms.ToTensor()])

train_images = ImageDataset(image_root=train_images_path, label_root=train_labels_path, transform=transforms)
val_images = ImageDataset(image_root=val_images_path, label_root=val_labels_path, transform=transforms)

train_loader = torch.utils.data.DataLoader(train_images, batch_size=batch_size, shuffle=True)
model = Model()
with torch.no_grad():
    x, y = next(iter(train_loader))
    print(f'x = {x}')
    print(f'y_true = {y}')
    print(f'y_pred = {model(x)}')

# проверка коректности модели
summary(model, (3, 448, 448), device="cpu")