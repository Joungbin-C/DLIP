//
// Created by joung on 25. 4. 15.
//

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat Background_image;
Mat video;
int hmin = 50, hmax = 100, smin = 60, smax = 255, vmin = 90, vmax = 255; // Sample 2

int main()
{
    Mat video_disp, video_hsv, video_erase;
    vector<vector<Point> > contours;

    /*  open the video  */
    VideoCapture cap("sample2-1.mp4");
    // VideoCapture cap("sample2-2.mp4");
    if (!cap.isOpened())
    {
        cout << "Cannot open the video cam/n";
        return -1;
    }

    /*  open Background Image  */
    cap.read(Background_image);

    namedWindow("Source", 0);

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
        double threshold_Area = 1500;
        if (contours.size() > 0) {
            for (int i = 0; i < contours.size(); i++)
                if (contourArea(contours[i])>threshold_Area) {
                    drawContours(white_window, contours, i, Scalar(0, 0, 0), FILLED); //  Drawing the Contour Box
                }

            /// Operating bitwise to get object and background image ///
            Mat video_object = white_window & video;
            Mat video_background = ~white_window & Background_image;

            /// The result ///
            video_disp = video_background | video_object;

        }

        imshow("Source", video_disp);
        if (waitKey(10) == 27)
            break;
    }
    cap.release();
    destroyAllWindows();
}