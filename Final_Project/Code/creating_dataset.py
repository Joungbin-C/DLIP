import cv2 as cv                         # OpenCV
import os                                # For file management
from tqdm import tqdm                    # For progress bar
from PIL import Image                    # To handle images with PIL
import torchvision.transforms as T       # For image transformations
import torch                             # PyTorch core library
import torch.nn as nn                    # Neural network modules
import torch.nn.functional as F          # Functional operations like grid_sample
from torchinfo import summary            # For model summary
from torchvision.models import vgg16     # Pretrained VGG16 model
import numpy as np                       # Numpy

# Function to extract frames from a video file
def extract_frames_from_video(video_path, output_dir, prefix='video', every_nth=1):
    os.makedirs(output_dir, exist_ok=True)                     # Check output directory exists

    cap = cv.VideoCapture(video_path)                          # Open video file
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}") # Error if video fails to open

    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))              # Get total number of frames of video
    idx = 0
    saved = 0

    for i in tqdm(range(total), desc=f"Extracting frames from {video_path}"):
        ret, frame = cap.read()                                # Read a frame
        if not ret:
            break
        if i % every_nth != 0:                                 # Skip frames based on 'every_nth'
            continue

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)       # Convert BGR to RGB
        frame_path = os.path.join(output_dir, f"{prefix}_{saved:05d}.png")  # Output filename
        cv.imwrite(frame_path, cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR))    # Save frame
        saved += 1

    cap.release()                                              # Release the video
    print(f"Saved {saved} frames to: {output_dir}")            # Print result

# Transformation pipeline for image preprocessing
transform = T.Compose([
    T.ToTensor(),                                              # Convert PIL image to tensor
])

device = "cuda" if torch.cuda.is_available() else "cpu"        # Choose GPU if available
print(f"Using {device} device")

# Function to prepare input image pairs (Xa/Xb)
def prepare_xa_xb_pairs(frame_dir, xa_dir, xb_dir, step=1):
    os.makedirs(xa_dir, exist_ok=True)                         # Create Xa output directory
    os.makedirs(xb_dir, exist_ok=True)                         # Create Xb output directory

    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])  # Get sorted frame list

    for i in tqdm(range(len(frame_files) - step), desc="Creating Xa/Xb pairs"):
        src_a = os.path.join(frame_dir, frame_files[i])        # Path for frame i
        src_b = os.path.join(frame_dir, frame_files[i + step]) # Path for frame i + step

        img_a = Image.open(src_a).convert("RGB")               # Load and convert image A
        img_b = Image.open(src_b).convert("RGB")               # Load and convert image B

        tensor_a = transform(img_a)                            # Transform to tensor
        tensor_b = transform(img_b)

        img_a_resized = T.ToPILImage()(tensor_a)               # Convert tensor back to PIL image
        img_b_resized = T.ToPILImage()(tensor_b)

        dst_a = os.path.join(xa_dir, f"xa_{i:05d}.png")        # Destination path for Xa
        dst_b = os.path.join(xb_dir, f"xb_{i:05d}.png")        # Destination path for Xb

        img_a_resized.save(dst_a)                              # Save Xa image
        img_b_resized.save(dst_b)                              # Save Xb image

    print(f"Saved {len(frame_files) - step} resized and transformed Xa/Xb pairs.")

# Function to generate Y images using optical flow
def generate_Y_with_optical_flow(xa_dir, xb_dir, y_dir, sf):
    os.makedirs(y_dir, exist_ok=True)                          # Create output Y directory
    device = "cuda" if torch.cuda.is_available() else "cpu"    # Choose GPU if available

    xa_files = sorted(os.listdir(xa_dir))                      # List Xa images
    xb_files = sorted(os.listdir(xb_dir))                      # List Xb images

    # Parameters for Farneback optical flow
    flow_params = dict(
        pyr_scale=0.5, levels=3, winsize=60, iterations=10,
        poly_n=7, poly_sigma=1.5, flags=0
    )

    num_pairs = min(len(xa_files), len(xb_files))              # Use minimum number of pairs

    for i in tqdm(range(num_pairs), desc="Generating Y with Optical Flow"):
        path_a = os.path.join(xa_dir, xa_files[i])             # Load Xa image
        path_b = os.path.join(xb_dir, xb_files[i])             # Load Xb image

        img_a_np = np.array(Image.open(path_a).convert("RGB")) # Convert Xa to numpy array
        img_b_np = np.array(Image.open(path_b).convert("RGB")) # Convert Xb to numpy array
        gray_a = cv.cvtColor(img_a_np, cv.COLOR_RGB2GRAY)      # Convert to grayscale
        gray_b = cv.cvtColor(img_b_np, cv.COLOR_RGB2GRAY)

        H, W = gray_a.shape                                    # Image dimensions

        flow = cv.calcOpticalFlowFarneback(gray_a, gray_b, None, **flow_params)  # Compute optical flow

        flow_tensor = torch.from_numpy(flow).unsqueeze(0).to(device)  # Convert flow to torch tensor

        # Generate base grid
        xx = torch.arange(0, W, device=device).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, device=device).view(-1, 1).repeat(1, W)
        xx = 2.0 * xx / (W - 1) - 1.0
        yy = 2.0 * yy / (H - 1) - 1.0
        base_grid = torch.stack([xx, yy], dim=2).unsqueeze(0)

        # Scale flow for grid sampling
        scaled_flow = torch.zeros_like(flow_tensor)
        scaled_flow[..., 0] = flow_tensor[..., 0] / (W - 1) * 2.0
        scaled_flow[..., 1] = flow_tensor[..., 1] / (H - 1) * 2.0

        new_grid = base_grid + scaled_flow * sf                # Add scaled flow to base grid

        # Convert RGB image to tensor and normalize
        img_a_tensor = torch.from_numpy(img_a_np).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

        y = F.grid_sample(img_a_tensor, new_grid, mode='bilinear', padding_mode='border', align_corners=True)  # Apply warping

        y_img = T.ToPILImage()(y.squeeze(0).cpu())             # Convert output tensor to PIL image
        y_img.save(os.path.join(y_dir, f"y_{i:05d}.png"))      # Save Y image

    print(f"Saved {num_pairs} warped Y images using Optical Flow to {y_dir}")  # Completion message
