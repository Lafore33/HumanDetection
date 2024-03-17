import torch
from torch import nn

# [out_channels, kernel_size, stride, padding]
params = [
    (64, 7, 2, 3),
    "M",
    (192, 3, 1, 1),
    "M",
    (128, 1, 1, 0),
    (256, 3, 1, 1),
    (256, 1, 1, 0),
    (512, 3, 1, 1),
    "M",
    [(256, 1, 1, 0), (512, 3, 1, 1), 4],
    (512, 1, 1, 0),
    (1024, 3, 1, 1),
    "M",
    [(512, 1, 1, 0), (1024, 3, 1, 1), 2],
    (1024, 3, 1, 1),
    (1024, 3, 2, 1),
    (1024, 3, 1, 1),
    (1024, 3, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        return x


class YOLO(nn.Module):
    def __init__(self, config, in_channels=3, cells_amount=7, num_boxes=2, num_classes=1):
        super(YOLO, self).__init__()
        self.in_channels = in_channels
        self.architecture = config
        self.conv_block = self.create_conv_block(self.architecture)
        self.fc_block = self.create_fc_block(cells_amount, num_boxes, num_classes)

    def forward(self, x):
        x = self.conv_block(x)
        x = torch.flatten(x, 1)
        x = self.fc_block(x)
        return x

    def create_conv_block(self, architecture):
        in_channels = self.in_channels
        layers = []
        for layer in architecture:
            if isinstance(layer, tuple):
                layers += [CNNBlock(in_channels, layer[0], layer[1], layer[2], layer[3])]
                in_channels = layer[0]

            elif isinstance(layer, str):
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            elif isinstance(layer, list):
                for _ in range(layer[2]):
                    layers += [CNNBlock(in_channels, layer[0][0], layer[0][1], layer[0][2], layer[0][3])]
                    in_channels = layer[0][0]
                    layers += [CNNBlock(in_channels, layer[1][0], layer[1][1], layer[1][2], layer[1][3])]
                    in_channels = layer[1][0]

        return nn.Sequential(*layers)

    def create_fc_block(self, cells_amount, num_boxes, num_classes):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * cells_amount * cells_amount, 496),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(496, cells_amount * cells_amount * (num_classes + num_boxes * 5))
        )
