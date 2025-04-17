//
// Created by joung on 25. 4. 11.
//

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat Background_image, video;
int hmin = 13, hmax = 18, smin = 108, smax = 150, vmin = 130, vmax = 230;

int main()
{
    Mat video_disp, video_hsv, video_erase;
    vector<vector<Point> > contours;

    /*  open the video  */
    VideoCapture cap("LAB_MagicCloak_Sample1.mp4");
    if (!cap.isOpened())
    {
        cout << "Cannot open the video cam\n";
        return -1;
    }

    /*  open Background Image  */


    while (true) {
        /*  capture the frame of video  */
        bool bSuccess = cap.read(video);
        if (!bSuccess)
        {
            cout << "Cannot find a frame from  video stream\n";
            break;
        }

        /// Final video and white window ///
        video.copyTo(video_disp);
        Mat white_window = Mat::zeros(video.size(), CV_8UC3) + Scalar(255,255,255);

        /// convert BGR to HSV ///
        cvtColor(video, video_hsv, COLOR_BGR2HSV);

        /// Find the area which is in the desirable range ///
        inRange(video_hsv, Scalar(MIN(hmin, hmax), MIN(smin, smax), MIN(vmin, vmax)), Scalar(MAX(hmin, hmax), MAX(smin, smax), MAX(vmin, vmax)), video_erase);

        /// Dilating to decrease the noise ///
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(13, 13));
        dilate(video_erase, video_erase, kernel);

        /// Find the contours ///
        findContours(video_erase, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

        /// Find the Contour with the largest area ///
        double threshold_Area = 50;
        double maxArea = 0;
        int maxArea_idx = 0;
        if (contours.size() > 0) {
            for (int i = 0; i < contours.size(); i++)
                if (contourArea(contours[i]) > maxArea && contourArea(contours[i]) > threshold_Area) {
                    maxArea = contourArea(contours[i]);
                    maxArea_idx = i;
                }

            ///  Drawing the Contour Box  ///
            drawContours(white_window, contours, maxArea_idx, Scalar(0, 0, 0), FILLED);

            /// Operating bitwise to get object and background image ///
            Mat video_object = white_window & video;
            Mat video_background = ~white_window & Background_image;

            /// The result ///
            video_disp = video_background | video_object;
            imshow("Source", video_disp);
        }

        if (waitKey(10) == 27)
            break;
    }
    cap.release();
    destroyAllWindows();
}