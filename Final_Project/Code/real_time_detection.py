# ==== Import Libraries ====
import cv2 as cv                                       # OpenCV for video capture and image processing
import torch                                           # PyTorch for deep learning operations
import torchvision.transforms as T                     # For preprocessing image tensors
from PIL import Image                                  # PIL for handling image format conversion
import numpy as np                                     # NumPy for numerical array operations
from magnification_model import MagnificationModel     # Your custom model for motion magnification

# ==== Configuration ====
SF = 20.0                                              # Magnification scale factor
FRAME_SIZE = (500, 500)                                # Resize all frames to this resolution
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, else CPU
SOURCE = "video/speed120.mp4"                          # Video file source (not used in this version)
MAX_FRAMES = 50                                        # Max number of frames to store for XT image creation

# ==== Load the pre-trained magnification model ====
def load_magnification_model(path, device):
    model = MagnificationModel()                               # Instantiate model
    model.load_state_dict(torch.load(path, map_location=device))  # Load trained weights
    model.to(device)                                           # Move model to appropriate device
    model.eval()                                               # Set model to evaluation mode
    return model

# ==== Preprocessing: Convert OpenCV frame to tensor ====
transform = T.Compose([
    T.Resize(FRAME_SIZE),                      # Resize image
    T.ToTensor()                               # Convert to tensor and normalize to [0, 1]
])

def preprocess(frame):
    img = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))  # Convert BGR frame to RGB PIL image
    return transform(img).unsqueeze(0).to(DEVICE)                # Apply transform, add batch dim, move to device

# ==== Postprocessing: Convert model tensor back to BGR image ====
def postprocess(tensor):
    tensor = tensor.squeeze(0).clamp(0, 1)                       # Remove batch dimension, clamp values
    np_img = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # Convert to HWC format, scale to 0-255
    return cv.cvtColor(np_img, cv.COLOR_RGB2BGR)                 # Convert RGB to BGR for OpenCV

# ==== Extract a horizontal slice from a grayscale image (for XT visualization) ====
def extract_xt_slice(gray_img, y_mid, x_start, x_end, crop_height):
    horizontal_slice = gray_img[y_mid - crop_height//2 : y_mid + crop_height//2 + 1, x_start:x_end]
    return np.median(horizontal_slice, axis=0)                   # Take median over height to reduce noise

# ==== Build 2D XT image from multiple 1D slices ====
def process_xt_images(xt_buffer_input, xt_buffer_magnified, w=500, h=500):
    xt_input = np.stack(xt_buffer_input, axis=0)                 # Stack all input slices (shape: [T, X])
    xt_magnified = np.stack(xt_buffer_magnified, axis=0)         # Stack all magnified slices

    xt_input_norm = cv.normalize(xt_input, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    xt_magnified_norm = cv.normalize(xt_magnified, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    xt_input_rot = cv.rotate(xt_input_norm, cv.ROTATE_90_COUNTERCLOCKWISE)    # Rotate for visualization
    xt_magnified_rot = cv.rotate(xt_magnified_norm, cv.ROTATE_90_COUNTERCLOCKWISE)

    xt_input_color = cv.cvtColor(cv.resize(xt_input_rot, (w, h)), cv.COLOR_GRAY2BGR)       # Grayscale to color
    xt_magnified_color = cv.cvtColor(cv.resize(xt_magnified_rot, (w, h)), cv.COLOR_GRAY2BGR)

    return xt_input_color, xt_magnified_color

# ==== Analyze peaks (vibrations) in the XT image ====
def analyze_peaks(xt_img_color, baseline_y=300):
    roi = xt_img_color[0:baseline_y, :]                          # Region of interest above baseline
    gray_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)               # Convert to grayscale
    _, binary_roi = cv.threshold(gray_roi, 125, 255, cv.THRESH_BINARY)  # Binarize
    edges_roi = cv.Canny(binary_roi, 50, 150)                    # Detect edges

    contours, _ = cv.findContours(edges_roi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # Find contours
    contour_result = roi.copy()
    cv.drawContours(contour_result, contours, -1, (0, 255, 0), 1)  # Draw all contours in green

    peak_count = 0
    for cnt in contours:
        peak = tuple(cnt[cnt[:, :, 1].argmin()][0])              # Find topmost point in each contour
        peak_count += 1
        cv.circle(contour_result, peak, 3, (255, 0, 0), -1)       # Mark peak with blue dot

    cv.putText(contour_result, f"Peaks above baseline: {peak_count}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)      # Annotate number of peaks
    return contour_result, peak_count

# ==== Main loop for real-time vibration detection and visualization ====
def run_detection_loop(video_path, model, sf_value=20.0,
                       output_combined="output_combined_80.mp4",
                       output_contour="output_contour_80.mp4"):

    cap = cv.VideoCapture(1)                            # Open webcam (use 0 or 1 for device index)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera or video")

    xt_buffer = []                                      # Buffer to store XT slices from magnified frames
    xt_buffer_input = []                                # Buffer to store XT slices from input frames
    prev_tensor = None                                  # To store the previous frame as tensor
    h, w = FRAME_SIZE

    fps = cap.get(cv.CAP_PROP_FPS)                      # Get frame rate
    out_h = h * 2                                       # Height of final visualization window
    out_w = w * 2                                       # Width of final visualization window
    fourcc = cv.VideoWriter_fourcc(*'mp4v')             # Video codec
    combined_writer = cv.VideoWriter(output_combined, fourcc, fps, (out_w, out_h))
    contour_writer = cv.VideoWriter(output_contour, fourcc, fps, (out_w, out_h))

    while True:
        ret, frame = cap.read()                         # Read frame from camera
        if not ret:
            break

        tensor = preprocess(frame)                      # Preprocess current frame

        if prev_tensor is not None:                     # Only start after having two frames
            with torch.no_grad():
                y_hat = model(prev_tensor, tensor, torch.tensor(sf_value).to(DEVICE))  # Magnify motion
                out_frame = postprocess(y_hat)          # Convert magnified output to image
                in_frame = postprocess(tensor)          # Convert input to image

            gray_mag = cv.cvtColor(out_frame, cv.COLOR_BGR2GRAY)   # Grayscale magnified
            gray_inp = cv.cvtColor(in_frame, cv.COLOR_BGR2GRAY)    # Grayscale input

            y_mid = h // 2                              # Vertical center of the frame
            x_start, x_end, crop_height = w // 2 + 50, w // 2 + 80, 30  # XT crop region (slightly right of center)

            xt_buffer.append(extract_xt_slice(gray_mag, y_mid, x_start, x_end, crop_height))
            xt_buffer_input.append(extract_xt_slice(gray_inp, y_mid, x_start, x_end, crop_height))

            if len(xt_buffer) > MAX_FRAMES:             # Keep buffer size fixed
                xt_buffer.pop(0)
                xt_buffer_input.pop(0)

            xt_input_color, xt_magnified_color = process_xt_images(xt_buffer_input, xt_buffer)

            frame_disp = cv.resize(frame, FRAME_SIZE)
            out_frame_disp = cv.resize(out_frame, FRAME_SIZE)
            cv.line(out_frame_disp, (x_start, y_mid), (x_end, y_mid), (255, 0, 0), 2)  # Mark XT crop region

            top_row = np.hstack((frame_disp, out_frame_disp))         # Combine input + output
            bottom_row = np.hstack((xt_input_color, xt_magnified_color))  # Combine XT visualizations
            combined = np.vstack((top_row, bottom_row))               # Full 2x2 view

            contour_result, peak_count = analyze_peaks(xt_magnified_color)

            # Display warning messages depending on peak count
            if peak_count >= 13:
                cv.putText(combined, "Emergency! Stop the Machine!", (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv.putText(combined, "7.75Hz", (10, 60),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif 10 < peak_count < 13:
                cv.putText(combined, "Warning! Possible Breakdown", (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv.putText(combined, "6.63Hz", (10, 60),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                cv.putText(combined, "Normal State", (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv.putText(combined, "4.45Hz", (10, 60),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            contour_resized = cv.resize(contour_result, (out_w, out_h))
            # contour_writer.write(contour_resized)       # Save contour output if needed
            # combined_writer.write(combined)             # Save combined output if needed

            cv.imshow("Frame", contour_result)            # Show XT contour visualization
            cv.imshow("Result", combined)                 # Show full dashboard

        prev_tensor = tensor                              # Store current frame for next loop

        if cv.waitKey(1) & 0xFF == ord('q'):              # Press 'q' to quit
            break

    # ==== Release resources ====
    cap.release()
    combined_writer.release()
    contour_writer.release()
    cv.destroyAllWindows()
