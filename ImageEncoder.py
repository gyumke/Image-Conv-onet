import torch
import torch.nn as nn
import torch.nn.functional as F
from convolutional_occupancy_networks.src.encoder.unet import UNet

class LocalImageEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size =3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size =3, padding=1)
       # self.pooling = nn.AvgPool2d(2, 2)
        self.unet = UNet(64, in_channels=64, depth=4, merge_mode = "concat").to(device)

    def forward(self, x):
        x = x.float()
        c = F.relu(self.conv1(x))
        c = F.relu(self.conv2(c))

        #c = self.pooling(c)

        c = self.unet(c)

        return {"xz" : c}