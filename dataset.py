import numpy as np
import torch
import os
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from model import YOLO
from torchsummary import summary
from torch import nn
import cv2

CELLS_SPLIT = 7
NUM_BOXES = 2
NUM_CLASSES = 1


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, image_root, label_root, cells_split=CELLS_SPLIT,
                 num_boxes=NUM_BOXES, num_classes=NUM_CLASSES, transform=None):
        self.image_root = image_root
        self.label_root = label_root
        self.size = len(os.listdir(image_root))
        self.transform = transform
        self.cells_split = cells_split
        self.num_boxes = num_boxes
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image_file = f"img{index}.jpg"
        image_path = os.path.join(self.image_root, image_file)
        img = Image.open(image_path)
        if self.transform is not None:
            img = self.transform(img)

        label_file = image_file[:-3] + "txt"
        label_path = os.path.join(self.label_root, label_file)

        label = torch.tensor(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2))
        label_matrix = torch.zeros((self.cells_split, self.cells_split, self.num_boxes * 5 + self.num_classes))
        for box in label:
            class_label, x, y, height, width = box
            class_label = int(class_label)
            hc, vc = int(self.cells_split * x), int(self.cells_split * y)
            x_cell, y_cell = self.cells_split * x - hc, self.cells_split * y - vc
            height_cell, width_cell = self.cells_split * height, self.cells_split * width

            if label_matrix[vc, hc, self.num_classes] == 0:
                label_matrix[vc, hc, self.num_classes] = 1
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[vc, hc, self.num_classes + 1: self.num_classes + 5] = box_coordinates
                label_matrix[vc, hc, class_label] = 1

        return img, label_matrix


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

