import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from PIL import Image

CELLS_SPLIT = 7
NUM_BOXES = 1
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

    def __getitem__(self, idx):
        prefix = f"img{idx}"
        image_file = prefix + ".jpg"
        label_file = prefix + ".txt"

        image_path = os.path.join(self.image_root, image_file)
        img = Image.open(image_path)
        if self.transform is not None:
            img = self.transform(img)
        label_path = os.path.join(self.label_root, label_file)
        label_matrix = torch.zeros((self.cells_split, self.cells_split, self.num_boxes * 4 + self.num_classes))
        if os.path.getsize(label_path) != 0:
            label = torch.tensor(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2))
        else:
            return img, label_matrix
        # the intuition here is that each cell predicts NUM_BOXES, that's the parameter we set ourselves;
        # one box has 5 parameters [probability of presence of the obj, x, y, width, height] + num_of classes

        for box in label:
            class_label, x, y, width, height = box
            class_label = int(class_label)
            # here we get vertical and horizontal number of cells for one particular box
            hc, vc = int(self.cells_split * x), int(self.cells_split * y)

            # coordinates with respect to cell
            x_cell, y_cell = self.cells_split * x - hc, self.cells_split * y - vc
            height_cell = self.cells_split * height
            width_cell = self.cells_split * width

            # check whether this box for this class is presented; the problem here is that model can detect only one
            # object in a cell, so if we already have object of some class in this cell, we won't add another one
            if label_matrix[vc, hc, class_label] == 0:
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[vc, hc, self.num_classes: self.num_classes + 5] = box_coordinates
                label_matrix[vc, hc, class_label] = 1

        return img, label_matrix

