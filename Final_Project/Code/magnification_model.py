# ==== Library imports ====
import cv2 as cv                           # OpenCV for video/image handling
import os                                 # For directory operations
from tqdm import tqdm                     # Progress bar
from PIL import Image                     # PIL for image processing
import torchvision.transforms as T        # Image transformation utilities
import torch                              # PyTorch core library
import torch.nn as nn                     # Neural network module
import torch.nn.functional as F           # Functional operations (e.g., activations, pooling)
from torchinfo import summary             # Model summary info
from torchvision.models import vgg16      # Pretrained VGG16 (not used here)
import numpy as np                        # NumPy for numerical operations

# ==== Device configuration ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Video path and output directories ====
video_path = "video/speed80.mp4"
output_dir = "data/frames_sample/"
frame_dir = "data/frames_sample/L"
xa_dir = "data/training_data/L/Xa"
xb_dir = "data/training_data/L/Xb"

# ==== Residual Block ====
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x                            # Save input for residual connection
        out = self.relu(self.conv1(x))          # First conv + ReLU
        out = self.conv2(out)                   # Second conv
        return self.relu(out + residual)        # Add skip connection and ReLU

# ==== Encoder: Extract shape and texture representations ====
class Encoder(nn.Module):
    def __init__(self, in_channels=3, mid_channels=16, out_channels=32):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

        self.downsample = nn.Sequential(self.conv1, self.relu, self.conv2, self.relu)
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
        x = self.downsample(x)                   # Initial downsampling
        x = self.resblocks3(x)                   # Shared residual blocks
        M = self.shape_head(x)                   # Shape feature output
        V = self.texture_head(x)                 # Texture feature output (more downsampled)
        return M, V

# ==== Manipulator: Apply motion magnification ====
class Manipulator(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.g = nn.Sequential(                  # Optional block (not used here)
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.h = nn.Sequential(                  # Optional block (not used here)
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            ResidualBlock(channels)
        )

    def forward(self, Ma, Mb, SF):
        diff = Mb - Ma                           # Temporal difference between frames
        B, C, H, W = diff.shape

        if SF.dim() == 1:
            SF = SF.view(B, 1, 1, 1)             # Expand SF shape for broadcasting

        return Ma + SF * diff                    # Apply magnification to shape features

# ==== Decoder: Reconstruct image from features ====
class Decoder(nn.Module):
    def __init__(self, shape_channels=32, texture_channels=32, mid_channels=64, out_channels=3):
        super(Decoder, self).__init__()

        self.upsample_texture = nn.Upsample(scale_factor=2, mode='nearest')  # Upsample texture map

        self.merge = nn.Sequential(              # Merge shape and texture
            nn.Conv2d(texture_channels + shape_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(mid_channels)
        )

        self.res_blocks = nn.Sequential(*[       # Processing after merge
            ResidualBlock(mid_channels) for _ in range(5)
        ])

        self.upsample_final = nn.Sequential(     # Final upsample to original resolution
            nn.ConvTranspose2d(mid_channels, shape_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(shape_channels, out_channels, kernel_size=7, padding=3)
        self.output_act = nn.Sigmoid()           # Normalize output to [0, 1]

    def forward(self, texture_repr, shape_repr):
        texture = self.upsample_texture(texture_repr)             # Upsample texture to match shape
        x = torch.cat([texture, shape_repr], dim=1)               # Concatenate features
        x = self.merge(x)                                         # Merge and normalize
        x = self.res_blocks(x)                                    # Deep processing
        x = self.upsample_final(x)                                # Upsample to original scale
        out = self.final_conv(x)                                  # Final output image
        return self.output_act(out)                               # Sigmoid activation

# ==== Full Model: MagnificationModel ====
class MagnificationModel(nn.Module):
    def __init__(self):
        super(MagnificationModel, self).__init__()
        self.encoder = Encoder()                  # Feature encoder
        self.manipulator = Manipulator()          # Magnification unit
        self.decoder = Decoder()                  # Image reconstruction

    def forward(self, Xa, Xb, SF):
        Ma, Va = self.encoder(Xa)                 # Encode frame A
        Mb, Vb = self.encoder(Xb)                 # Encode frame B

        M_manipulated = self.manipulator(Ma, Mb, SF)  # Apply SF magnification
        Y_hat = self.decoder(Va, M_manipulated)       # Decode to image
        return Y_hat

# ==== Model test and summary ====
model = MagnificationModel().to(device)           # Instantiate model and move to device

input_Xa = torch.randn(1, 3, 224, 224).to(device)  # Dummy input: frame A
input_Xb = torch.randn(1, 3, 224, 224).to(device)  # Dummy input: frame B
SF = torch.tensor(20.0).to(device)                # Magnification scale factor

# Print model summary
summary(model, input_data=(input_Xa, input_Xb, SF),
        col_names=["input_size", "output_size", "num_params"],
        depth=3)
