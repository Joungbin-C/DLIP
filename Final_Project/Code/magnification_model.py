import cv2 as cv
import os
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import torch
from torchvision.models import vgg16
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

video_path = "video/speed80.mp4"
output_dir = "data/frames_sample/"
frame_dir = "data/frames_sample/L"
xa_dir = "data/training_data/L/Xa"
xb_dir = "data/training_data/L/Xb"

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + residual)

class Encoder(nn.Module):
    def __init__(self, in_channels=3, mid_channels=16, out_channels=32):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

        self.downsample = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu
        )

        self.resblocks2 = nn.Sequential(*[ResidualBlock(out_channels) for _ in range(2)])
        self.resblocks3 = nn.Sequential(*[ResidualBlock(out_channels) for _ in range(3)])

        self.shape_head = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            self.resblocks2
        )
        self.texture_head = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            self.resblocks2
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.resblocks3(x)
        M = self.shape_head(x)       # Shape Representation
        V = self.texture_head(x)     # Texture Representation (extra 2x downsampled)
        return M, V

class Manipulator(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.g = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.h = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            ResidualBlock(channels)
        )

    def forward(self, Ma, Mb, SF):
        diff = Mb - Ma  # [B, C, H, W]
        B, C, H, W = diff.shape

        if SF.dim() == 1:
            SF = SF.view(B, 1, 1, 1)

        return Ma + SF * diff

class Decoder(nn.Module):
    def __init__(self, shape_channels=32, texture_channels=32, mid_channels=64, out_channels=3):
        super(Decoder, self).__init__()

        self.upsample_texture = nn.Upsample(scale_factor=2, mode='nearest')  # Texture: [B, 32, H/4, W/4] → [B, 32, H/2, W/2]

        self.merge = nn.Sequential(
            nn.Conv2d(texture_channels + shape_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(mid_channels)
        )

        self.res_blocks = nn.Sequential(*[
            ResidualBlock(mid_channels) for _ in range(5)
        ])

        self.upsample_final = nn.Sequential(
            nn.ConvTranspose2d(mid_channels, shape_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(shape_channels, out_channels, kernel_size=7, padding=3)
        self.output_act = nn.Sigmoid()  # [0, 1] 범위로 정규화

    def forward(self, texture_repr, shape_repr):
        texture = self.upsample_texture(texture_repr)             # [B, 32, H/2, W/2]
        x = torch.cat([texture, shape_repr], dim=1)               # [B, 64, H/2, W/2]
        x = self.merge(x)                                         # [B, 64, H/2, W/2]
        x = self.res_blocks(x)                                    # 5 ResBlocks
        x = self.upsample_final(x)                                # [B, 32, H, W]
        out = self.final_conv(x)                                  # [B, 3, H, W]

        # visualize_tensor(texture, title='Texture Feature')
        # visualize_tensor(shape_repr, title='Shape Feature')

        return self.output_act(out)                               # Sigmoid activation


class MagnificationModel(nn.Module):
    def __init__(self):
        super(MagnificationModel, self).__init__()
        self.encoder = Encoder()
        self.manipulator = Manipulator()
        self.decoder = Decoder()

    def forward(self, Xa, Xb, SF):
        Ma, Va = self.encoder(Xa)
        Mb, Vb = self.encoder(Xb)

        # Vb = F.avg_pool2d(Vb, kernel_size=3, stride=1, padding=1)

        # Shape manipulation using magnification factor
        M_manipulated = self.manipulator(Ma, Mb, SF)

        # Decode manipulated shape and texture
        Y_hat = self.decoder(Va, M_manipulated)
        return Y_hat

model = MagnificationModel().to(device)

input_Xa = torch.randn(1, 3, 224, 224).to(device)
input_Xb = torch.randn(1, 3, 224, 224).to(device)
SF = torch.tensor(20.0).to(device)

summary(model, input_data=(input_Xa, input_Xb, SF),
        col_names=["input_size", "output_size", "num_params"],
        depth=3)