import numpy as np
import cv2 as cv

cap = cv.VideoCapture('LAB3_Video.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Open the Image in Gray Scale and Filtering
    img = frame.copy()
    height, width = img.shape[:2]

    # Cropped the image in ROI
    roi_x = 0; roi_y = 400; roi_w = 580; roi_h = height
    roi = img[roi_y:roi_h, roi_x:roi_w]
    img2 = roi.copy()

    # Split the channel and convert the red to gray scale
    img_b, img_g, img_r = cv.split(img2)
    img_denoised = cv.bilateralFilter(img_r, d=9, sigmaColor=75, sigmaSpace=75)

    # Canny Edge Detection
    img_canny = cv.Canny(img_denoised, 50, 80)
    cv.imshow('denoised image', img_denoised)
    cv.imshow('canny image', img_canny)

    lines = cv.HoughLinesP(img_canny, 1, np.pi/180, 30, minLineLength=20, maxLineGap=10)
    X = []
    Y = []
    if lines is not None:
        for i in range(lines.shape[0]):
            x1 = lines[i][0][0] + roi_x
            y1 = lines[i][0][1]
            x2 = lines[i][0][2] + roi_x
            y2 = lines[i][0][3]
            cv.line(img2, (x1, y1), (x2, y2), (0, 255, 255), 1, cv.LINE_AA)
            X.extend([x1, x2])
            Y.extend([y1, y2])

        # Curve Fitting
        A = np.vstack([np.power(X, 2), X, np.ones(len(X))]).T
        Y_np = np.array(Y).reshape(-1, 1)
        ATA = A.T @ A
        ATY = A.T @ Y_np
        coeffs = np.linalg.inv(ATA) @ ATY

        x = np.arange(0, width)
        y = coeffs[0] * x ** 2 + coeffs[1] * x + coeffs[2] + roi_y

        mask_level1 = y > (height - 250)
        mask_level2 = y > (height - 120)
        under_x_250px = x[mask_level1]
        under_x_120px = x[mask_level2]

        points = np.array(list(zip(x.astype(int), y.astype(int))), dtype=np.int32)
        cv.polylines(img, [points.reshape((-1, 1, 2))], False, (0, 255, 0), 2, cv.LINE_AA)
        cv.imshow('HoughLinesP', img2)

        # Minimum Value
        a, b, c = coeffs
        x_min = -b / (2 * a)
        y_min = a * x_min ** 2 + b * x_min + c + roi_y
        bottom_point = (x_min, y_min)

        # Calculate Score and Level
        score = roi_h - y_min
        if score < 120:
            level = 3
            if under_x_250px.size > 0:
                cv.line(img, (0, roi_h - 250), (under_x_250px[0], roi_h - 250), (0, 0, 255), 2, cv.LINE_AA)
                cv.line(img, (under_x_250px[-1], height - 250), (width, roi_h - 250), (0, 0, 255), 2, cv.LINE_AA)
            if under_x_120px.size > 0:
                cv.line(img, (0, roi_h - 120), (under_x_120px[0], roi_h - 120), (255, 0, 0), 2, cv.LINE_AA)
                cv.line(img, (under_x_120px[-1], roi_h - 120), (width, roi_h - 120), (255, 0, 0), 2, cv.LINE_AA)
        elif 120 <= score < 250:
            level = 2
            if under_x_250px.size > 0:
                cv.line(img, (0, roi_h - 250), (under_x_250px[0], roi_h - 250), (0, 0, 255), 2, cv.LINE_AA)
                cv.line(img, (under_x_250px[-1], height - 250), (width, roi_h - 250), (0, 0, 255), 2, cv.LINE_AA)
            cv.line(img, (0, roi_h - 120), (width, roi_h - 120), (255, 0, 0), 2, cv.LINE_AA)
        elif score >= 250:
            level = 1
            cv.line(img, (0, roi_h - 250), (width, roi_h - 250), (0, 0, 255), 2, cv.LINE_AA)
            cv.line(img, (0, roi_h - 120), (width, roi_h - 120), (255, 0, 0), 2, cv.LINE_AA)

    else:
        print('No lines detected')

    box_x, box_y = 880, 350
    box_w, box_h = 240, 80
    overlay = img.copy()
    alpha = 0.5
    cv.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
    cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    cv.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), 2)

    cv.putText(img, f'Level: {level}', (box_x + 10, box_y + 35),
               cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(img, f'Score: {int(score)}', (box_x + 10, box_y + 70),
               cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv.LINE_AA)

    cv.namedWindow('Result', cv.WINDOW_NORMAL)
    cv.resizeWindow('Result', width=960, height=540)
    cv.imshow('Result', img)

    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()