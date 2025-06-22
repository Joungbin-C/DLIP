import cv2 as cv
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from magnification_model import MagnificationModel

# --- 설정 ---
SF = 20.0
FRAME_SIZE = (500, 500)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SOURCE = "video/speed120.mp4"
MAX_FRAMES = 50

# Magnification 모델 로드
def load_magnification_model(path, device):
    model = MagnificationModel()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 전처리 및 후처리 함수
transform = T.Compose([
    T.Resize(FRAME_SIZE),
    T.ToTensor()
])

def preprocess(frame):
    img = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    return transform(img).unsqueeze(0).to(DEVICE)

def postprocess(tensor):
    tensor = tensor.squeeze(0).clamp(0, 1)
    np_img = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv.cvtColor(np_img, cv.COLOR_RGB2BGR)

def extract_xt_slice(gray_img, y_mid, x_start, x_end, crop_height):
    horizontal_slice = gray_img[y_mid - crop_height//2 : y_mid + crop_height//2 + 1, x_start:x_end]
    return np.median(horizontal_slice, axis=0)

def process_xt_images(xt_buffer_input, xt_buffer_magnified, w=500, h=500):
    xt_input = np.stack(xt_buffer_input, axis=0)
    xt_magnified = np.stack(xt_buffer_magnified, axis=0)

    xt_input_norm = cv.normalize(xt_input, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    xt_magnified_norm = cv.normalize(xt_magnified, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    xt_input_rot = cv.rotate(xt_input_norm, cv.ROTATE_90_COUNTERCLOCKWISE)
    xt_magnified_rot = cv.rotate(xt_magnified_norm, cv.ROTATE_90_COUNTERCLOCKWISE)

    xt_input_color = cv.cvtColor(cv.resize(xt_input_rot, (w, h)), cv.COLOR_GRAY2BGR)
    xt_magnified_color = cv.cvtColor(cv.resize(xt_magnified_rot, (w, h)), cv.COLOR_GRAY2BGR)

    return xt_input_color, xt_magnified_color

def analyze_peaks(xt_img_color, baseline_y=300):
    roi = xt_img_color[0:baseline_y, :]
    gray_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    _, binary_roi = cv.threshold(gray_roi, 125, 255, cv.THRESH_BINARY)
    edges_roi = cv.Canny(binary_roi, 50, 150)

    contours, _ = cv.findContours(edges_roi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour_result = roi.copy()
    cv.drawContours(contour_result, contours, -1, (0, 255, 0), 1)

    peak_count = 0
    for cnt in contours:
        peak = tuple(cnt[cnt[:, :, 1].argmin()][0])
        peak_count += 1
        cv.circle(contour_result, peak, 3, (255, 0, 0), -1)

    cv.putText(contour_result, f"Peaks above baseline: {peak_count}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    return contour_result, peak_count

def run_detection_loop(video_path, model, sf_value=20.0,
                       output_combined="output_combined_80.mp4",
                       output_contour="output_contour_80.mp4"):
    cap = cv.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("카메라/영상 열기 실패")

    xt_buffer = []
    xt_buffer_input = []
    prev_tensor = None
    h, w = FRAME_SIZE

    fps = cap.get(cv.CAP_PROP_FPS)
    out_h = h * 2
    out_w = w * 2
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    combined_writer = cv.VideoWriter(output_combined, fourcc, fps, (out_w, out_h))
    contour_writer = cv.VideoWriter(output_contour, fourcc, fps, (out_w, out_h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tensor = preprocess(frame)

        if prev_tensor is not None:
            with torch.no_grad():
                y_hat = model(prev_tensor, tensor, torch.tensor(sf_value).to(DEVICE))
                out_frame = postprocess(y_hat)
                in_frame = postprocess(tensor)

            gray_mag = cv.cvtColor(out_frame, cv.COLOR_BGR2GRAY)
            gray_inp = cv.cvtColor(in_frame, cv.COLOR_BGR2GRAY)

            y_mid = h // 2
            x_start, x_end, crop_height = w // 2 + 50, w // 2 + 80, 30

            xt_buffer.append(extract_xt_slice(gray_mag, y_mid, x_start, x_end, crop_height))
            xt_buffer_input.append(extract_xt_slice(gray_inp, y_mid, x_start, x_end, crop_height))
            if len(xt_buffer) > MAX_FRAMES:
                xt_buffer.pop(0)
                xt_buffer_input.pop(0)

            xt_input_color, xt_magnified_color = process_xt_images(xt_buffer_input, xt_buffer)

            frame_disp = cv.resize(frame, FRAME_SIZE)
            out_frame_disp = cv.resize(out_frame, FRAME_SIZE)

            cv.line(out_frame_disp, (x_start, y_mid), (x_end, y_mid), (255, 0, 0), 2)

            top_row = np.hstack((frame_disp, out_frame_disp))
            bottom_row = np.hstack((xt_input_color, xt_magnified_color))
            combined = np.vstack((top_row, bottom_row))

            contour_result, peak_count = analyze_peaks(xt_magnified_color)

            if peak_count >= 13:
                cv.putText(combined, "Emergency! Stop the Machine!", (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv.putText(combined, "7.75Hz", (10, 60),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif 10 < peak_count < 13:
                cv.putText(combined, "Warning! Possible Breakdown", (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv.putText(combined, "6.63", (10, 60),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                cv.putText(combined, "Normal State", (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv.putText(combined, "4.45Hz", (10, 60),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            contour_resized = cv.resize(contour_result, (out_w, out_h))
            #contour_writer.write(contour_resized)
            #combined_writer.write(combined)

            cv.imshow("Frame", contour_result)
            cv.imshow("Result", combined)

        prev_tensor = tensor

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    combined_writer.release()
    contour_writer.release()
    cv.destroyAllWindows()
