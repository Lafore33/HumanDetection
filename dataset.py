import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from PIL import Image
# !pip install -U albumentations
# import albumentations as A
# from model import YOLO
# from torchsummary import summary
# from torch import nn
# import cv2

CELLS_SPLIT = 7
NUM_BOXES = 1
NUM_CLASSES = 1


def convert_bbox(original_size, bbox, new_size=(448, 448)):
    img_width = original_size[0]
    img_height = original_size[1]
    resized_width = new_size[0]
    resized_height = new_size[1]
    class_label, x_center, y_center, width, height = bbox

    x_top_left = (x_center - width / 2) * img_width * (resized_width / img_width)
    y_top_left = (y_center - height / 2) * img_height * (resized_height / img_height)
    x_bottom_right = (x_center + width / 2) * img_width * (resized_width / img_width)
    y_bottom_right = (y_center + height / 2) * img_height * (resized_height / img_height)

    x_center = (x_top_left + x_bottom_right) / (2 * resized_width)
    y_center = (y_top_left + y_bottom_right) / (2 * resized_height)
    width = (x_bottom_right - x_top_left) / resized_width
    height = (y_bottom_right - y_top_left) / resized_height

    return class_label, x_center, y_center, width, height


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, image_root, label_root, cells_split=CELLS_SPLIT,
                 num_boxes=NUM_BOXES, num_classes=NUM_CLASSES, transform=None):
        self.image_root = image_root
        self.label_root = label_root
        self.size = 8
        self.transform = transform
        self.cells_split = cells_split
        self.num_boxes = num_boxes
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if index >= 8:
            return
        image_file = f"img{index}.jpg"
        image_path = os.path.join(self.image_root, image_file)
        img = Image.open(image_path)

        label_file = image_file[:-3] + "txt"
        label_path = os.path.join(self.label_root, label_file)

        label = torch.tensor(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2))
        # the intuition here is that each cell predicts NUM_BOXES, it's the parameter we set ourselves;
        # one box has 5 parameters [probability of box, x, y, width, height] + num_of classes

        # as far as there is only 1 class, the probability of class can be erased
        # 'cause the confidence shows the presence of the object in the cell,
        # and we have only one possible object(human)
        label_matrix = torch.zeros((self.cells_split, self.cells_split, self.num_boxes * 4 + self.num_classes))

        orig_shape = np.array(img).shape
        if self.transform is not None:
            img = self.transform(img)
        # here we are running through all boxes for one image and deciding which cell it belongs
        for box in label:
            class_label, x, y, width, height = convert_bbox(orig_shape, box)
            class_label = int(class_label)

            # here we get vertical and horizontal number of cells for one particular box
            hc, vc = int(self.cells_split * x), int(self.cells_split * y)

            # coordinates with respect to cell
            x_cell, y_cell = self.cells_split * x - hc, self.cells_split * y - vc
            # ?
            height_cell = self.cells_split * height
            width_cell = self.cells_split * width

            # check whether this box for this class is presented; the problem here is that model can detect only one
            # object in a cell, so if we already have object of some class in this cell, we won't add another one
            if label_matrix[vc, hc, self.num_classes] == 0:
                label_matrix[vc, hc, class_label] = 1
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[vc, hc, self.num_classes: self.num_classes + 4] = box_coordinates

        return img, label_matrix
