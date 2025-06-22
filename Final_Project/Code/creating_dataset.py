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

def extract_frames_from_video(video_path, output_dir, prefix='video', every_nth=1):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    idx = 0
    saved = 0

    for i in tqdm(range(total), desc=f"Extracting frames from {video_path}"):
        ret, frame = cap.read()
        if not ret:
            break
        if i % every_nth != 0:
            continue

        # BGR → RGB, 저장
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_path = os.path.join(output_dir, f"{prefix}_{saved:05d}.png")
        cv.imwrite(frame_path, cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR))
        saved += 1

    cap.release()
    print(f"Saved {saved} frames to: {output_dir}")

# 이미지 크기 및 정규화 설정
transform = T.Compose([
    T.ToTensor(),
])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def prepare_xa_xb_pairs(frame_dir, xa_dir, xb_dir, step=1):
    os.makedirs(xa_dir, exist_ok=True)
    os.makedirs(xb_dir, exist_ok=True)

    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])

    for i in tqdm(range(len(frame_files) - step), desc="Creating Xa/Xb pairs"):
        src_a = os.path.join(frame_dir, frame_files[i])
        src_b = os.path.join(frame_dir, frame_files[i + step])

        img_a = Image.open(src_a).convert("RGB")
        img_b = Image.open(src_b).convert("RGB")

        tensor_a = transform(img_a)  # shape: [3, 224, 224]
        tensor_b = transform(img_b)

        # PIL 이미지로 변환해서 저장
        img_a_resized = T.ToPILImage()(tensor_a)
        img_b_resized = T.ToPILImage()(tensor_b)

        dst_a = os.path.join(xa_dir, f"xa_{i:05d}.png")
        dst_b = os.path.join(xb_dir, f"xb_{i:05d}.png")

        img_a_resized.save(dst_a)
        img_b_resized.save(dst_b)

    print(f"Saved {len(frame_files) - step} resized and transformed Xa/Xb pairs.")

def generate_Y_with_optical_flow(xa_dir, xb_dir, y_dir, sf):
    os.makedirs(y_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    xa_files = sorted(os.listdir(xa_dir))
    xb_files = sorted(os.listdir(xb_dir))

    flow_params = dict(
        pyr_scale=0.5, levels=3, winsize=60, iterations=10,
        poly_n=7, poly_sigma=1.5, flags=0
    )

    num_pairs = min(len(xa_files), len(xb_files))

    for i in tqdm(range(num_pairs), desc="Generating Y with Optical Flow"):
        path_a = os.path.join(xa_dir, xa_files[i])
        path_b = os.path.join(xb_dir, xb_files[i])

        # 1. 이미지를 OpenCV(Numpy) 형식으로 로드하고 흑백으로 변환
        img_a_np = np.array(Image.open(path_a).convert("RGB"))
        img_b_np = np.array(Image.open(path_b).convert("RGB"))
        gray_a = cv.cvtColor(img_a_np, cv.COLOR_RGB2GRAY)
        gray_b = cv.cvtColor(img_b_np, cv.COLOR_RGB2GRAY)

        H, W = gray_a.shape

        # 2. 최적의 플로우 계산
        flow = cv.calcOpticalFlowFarneback(gray_a, gray_b, None, **flow_params)

        # 3. Numpy Flow를 PyTorch Tensor로 변환
        flow_tensor = torch.from_numpy(flow).unsqueeze(0).to(device) # [1, H, W, 2]

        # 4. 베이스 그리드 생성
        xx = torch.arange(0, W, device=device).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, device=device).view(-1, 1).repeat(1, W)
        xx = 2.0 * xx / (W - 1) - 1.0
        yy = 2.0 * yy / (H - 1) - 1.0
        base_grid = torch.stack([xx, yy], dim=2).unsqueeze(0)

        # 5. Flow 값을 그리드 좌표계에 맞게 스케일링
        scaled_flow = torch.zeros_like(flow_tensor)
        scaled_flow[..., 0] = flow_tensor[..., 0] / (W - 1) * 2.0
        scaled_flow[..., 1] = flow_tensor[..., 1] / (H - 1) * 2.0

        # 6. 새로운 그리드 생성
        new_grid = base_grid + scaled_flow * sf

        # 7. 원본 컬러 이미지를 Tensor로 변환하여 grid_sample 적용
        img_a_tensor = torch.from_numpy(img_a_np).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        y = F.grid_sample(img_a_tensor, new_grid, mode='bilinear', padding_mode='border', align_corners=True)

        # 8. 결과 저장
        y_img = T.ToPILImage()(y.squeeze(0).cpu())
        y_img.save(os.path.join(y_dir, f"y_{i:05d}.png"))

    print(f"Saved {num_pairs} warped Y images using Optical Flow to {y_dir}")