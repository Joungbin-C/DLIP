#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

/*--------------function declaration---------------*/
void Show_Image(const Mat& img, const string& title);
Mat Preprocessing(const string& img_path);
Mat Filling_Holes(const Mat& bw);
Mat Applying_Morphology(const Mat& fill);
float Finding_Diameter(Mat open);
vector<vector<Point>> FindAndFilter_Teeth_Contours(const Mat& teeth_img);
void Calculate_Teeth_Properties(const vector<vector<Point>>& contours, const Point2f& gear_center, int& normal_teeth, int& defected_teeth, double& total_area, vector<double>& lengths,  double& total_length);
void Draw_Teeth_Contours(Mat& contourImg, Mat& img, const vector<vector<Point>>& contours, const Point2f& gear_center, const vector<double>& lengths, double avg_length);
void Printing_Teeth_Results(int normal_teeth, int defected_teeth, double total_area, float diameter);
void Analyzing_Teeth(const Mat& fill, const Mat& open, Mat& img, float diameter);

/*---------------------main---------------------*/
int main() {
    string img_path = "../Image/Gear1.jpg";
    Mat img = imread(img_path);

    if (img.empty()) {
        cerr << "Error: Could not read the image!" << endl;
        return 1;
    }

    Mat bw = Preprocessing(img_path);
    Mat fill = Filling_Holes(bw);
    Mat open = Applying_Morphology(fill);
    float diameter = Finding_Diameter(open);
    Analyzing_Teeth(fill, open, img, diameter);

    return 0;
}

/*----------------function definition----------------*/
// The Function to show image one by one
void Show_Image(const Mat& img, const string& title) {
    namedWindow(title, WINDOW_AUTOSIZE);
    imshow(title, img);
    waitKey(0);
    destroyAllWindows();
}

// Preprocessing image to make it to binary image and optimize it with Otsu method
Mat Preprocessing(const string& img_path) {
    Mat img = imread(img_path);
    if (img.empty()) {
        cerr << "Error: Could not read the image!" << endl;
        exit(1);
    }

    Mat gray, bw;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    threshold(gray, bw, 0, 255, THRESH_BINARY | THRESH_OTSU);

    Show_Image(bw, "Segmented Image");
    return bw;
}

// Filling a middle hole of the gear
Mat Filling_Holes(const Mat& bw) {
    Mat floodfill = bw.clone();
    floodFill(floodfill, Point(0, 0), Scalar(255));

    Mat floodfill_inv;
    bitwise_not(floodfill, floodfill_inv);

    Mat fill = (bw | floodfill_inv);
    Show_Image(fill, "Filled Image");
    return fill;
}

// Using opening method to make it a circle
Mat Applying_Morphology(const Mat& fill) {
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(140, 140));
    Mat open;
    morphologyEx(fill, open, MORPH_OPEN, kernel);
    Show_Image(open, "Opened Region");
    return open;
}

// Find the diameter with minEnclosingCircle function
float Finding_Diameter(Mat open) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(open, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    float radius;

    for (int i = 0; i < contours.size(); i++) {
        Point2f center;
        minEnclosingCircle(contours[i], center, radius);
    }
    return radius * 2;
}

// Find the contours with teeth-only image. Filtered noise contour.
vector<vector<Point>> FindAndFilter_Teeth_Contours(const Mat& teeth_img) {
    vector<vector<Point>> contours, filtered_contours;
    vector<Vec4i> hierarchy;
    findContours(teeth_img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    double area_min_threshold = 20;
    for (const auto& contour : contours) {
        if (contourArea(contour) >= area_min_threshold) {
            filtered_contours.push_back(contour);
        }
    }
    return filtered_contours;
}

// Calculate length, area, center of teeth, number of normal and defected teeth
void Calculate_Teeth_Properties(const vector<vector<Point>>& contours, const Point2f& gear_center, int& normal_teeth, int& defected_teeth, double& total_area, vector<double>& lengths, double& total_length) {
    lengths.clear();
    double area_max_threshold = 950, area_min_threshold = 20;

    for (int i = 0; i < contours.size(); i++) {
        Moments moment_contour = moments(contours[i]);
        Point2f teeth_center(moment_contour.m10 / moment_contour.m00, moment_contour.m01 / moment_contour.m00);

        double area = contourArea(contours[i]);
        double length = norm(gear_center - teeth_center);

        total_length += length;
        total_area += area;

        if (area > area_max_threshold) {
            normal_teeth++;
            lengths.push_back(length);
        } else if (area > area_min_threshold) {
            defected_teeth++;
            lengths.push_back(length);
        }
    }
}

// Draw contour and put text next to it. Draw circle on defected teeth.
void Draw_Teeth_Contours(Mat& contourImg, Mat& img, const vector<vector<Point>>& contours, const Point2f& gear_center, const vector<double>& lengths, double avg_length) {
    double length_offset = 50;
    double area_max_threshold = 950, area_min_threshold = 20;

    for (int i = 0; i < contours.size(); i++) {
        Moments moment_contour = moments(contours[i]);
        Point2f teeth_center(moment_contour.m10 / moment_contour.m00, moment_contour.m01 / moment_contour.m00);
        double area = contourArea(contours[i]);
        Point2f direction = gear_center - teeth_center;
        Point2f normalized_direction = direction / lengths[i];
        Point2f text_position = gear_center - normalized_direction * (avg_length - length_offset);

        Size textsize = getTextSize(to_string((int)area), FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);
        text_position.y += textsize.height / 2;
        text_position.x -= textsize.width / 2;

        if (area > area_max_threshold) {
            drawContours(contourImg, contours, i, Scalar(0, 255, 0), 2);
            putText(contourImg, to_string((int)area), text_position, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
        } else if (area < area_max_threshold && area > area_min_threshold) {
            drawContours(contourImg, contours, i, Scalar(0, 0, 255), 2);
            putText(contourImg, to_string((int)area), text_position, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
            circle(img, teeth_center, 40, Scalar(0, 255, 255), 2, LINE_AA);
        }
    }
}

// Print out the statistic results
void Printing_Teeth_Results(int normal_teeth, int defected_teeth, double total_area, float diameter) {
    double total_teeth = normal_teeth + defected_teeth;
    double avg_area = total_teeth == 0 ? 0 : total_area / total_teeth;

    cout << "Total Teeth: " << total_teeth << endl;
    cout << "Avg Teeth Area: " << avg_area << endl;
    cout << "Normal Teeth: " << normal_teeth << endl;
    cout << "Defected Teeth: " << defected_teeth << endl;
    cout << "Diameter: " << diameter << endl;

    if (defected_teeth != 0) {
        cout << "Quality Failed" << endl;
    } else {
        cout << "Quality Pass" << endl;
    }
}

// Combine all processes.
void Analyzing_Teeth(const Mat& fill, const Mat& open, Mat& img, float diameter) {
    Moments gearMoments = moments(open);
    Point2f gear_center((gearMoments.m10 / gearMoments.m00), gearMoments.m01 / gearMoments.m00);

    Mat teeth_img = fill - open;
    Show_Image(teeth_img, "Teeth Image");

    vector<vector<Point>> contours = FindAndFilter_Teeth_Contours(teeth_img);

    int normal_teeth = 0, defected_teeth = 0;
    double total_area = 0;
    vector<double> lengths = {0};
    double total_length = 0;
    Calculate_Teeth_Properties(contours, gear_center, normal_teeth, defected_teeth, total_area, lengths, total_length);

    double avg_length = lengths.empty() ? 0 : total_length / lengths.size();

    Mat contourImg = Mat::zeros(teeth_img.size(), CV_8UC3);
    Draw_Teeth_Contours(contourImg, img, contours, gear_center, lengths, avg_length);

    Show_Image(contourImg, "Detected Contours with Areas");
    Show_Image(img, "Defected Teeth Marked");

    Printing_Teeth_Results(normal_teeth, defected_teeth, total_area, diameter);
}

