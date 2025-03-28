# LAB: Grayscale Image Segmentation

**Date:**  2025-Mar-25

**Author:**  Joungbin Choi 22200757

**Github:** repository link  (if available)

**Demo Video:** Youtube link (if available)

---



# Introduction

## 1. Objective

The objective of this project is to analyze the condition of gear teeth using image processing techniques. Using various image processing method, determine the quality of gears

**Goal**: Identify defected teeth based on area irregularities and display gear image with marks on defected teeth. 



## 2. Preparation

### Software Installation

- OpenCV 3.83,  Clion2024.3.5

- CUDA 12.6, cudatoolkit 12.6, C++



### Dataset

There are four images of gear. Gear1.jpg and Gear3.jpg are images of defected gear and others are good quality. 

**Dataset link:** [Download the test image](https://github.com/ykkimhgu/DLIP-src/blob/main/LAB_grayscale/Lab_GrayScale_TestImage.jpg)






# Algorithm

## 1. Overview

![flowchart](https://github.com/Joungbin-C/DLIP/blob/main/LAB/DLIP_LAB_Grayscale_Image_Segmentation_22200757/img/flowchart.jpg)



## 2. Procedure

### Filtering

Even though the original image is in gray scale, it is not binary. Converting it to binary gray image is necessary to utilize Otsu threshold method. After converting the original image to binary gray image, Otsu threshold method is proceed to get rid of the noise that are nearby edge of the gear. 



### Thresholding

After converting the original image to binary gray image, Otsu threshold method is proceed to get rid of the noise that are nearby edge of the gear. After it, `floodFill()` Functions is used to fill the middle circle of the gear before processing morphology. 



### Morphology

The reason of doing morphology is to delete the teethes and make a circle to subtract from the original image. To delete the teethes, the size of kernel is 120X120 pixels and opening method is used. Teeth-only image is able to get after subtracting circle image from the original image. 



### Find Contours

**`findContours()`**: Find the contours and stored each contour as a vector of points. 

**`contourArea()`**: Calculate the area of each contour. It gives a number of pixels in each contour.

**`drawContours()`**: Draw colored line on each contour. There are `Area_max_threshold=950` and `area_min_threshold=20` value which is standard value of normal teeth size. If the area of contour in certain `idx` is not over `Area_max_threshold`, it is a defected teeth and draw a contour with a red line. If it is over `Area_min_threshold`, it is a defected teeth and draw a contour with a red line. Otherwise, that contour is normal teeth with a green line.

**Exception Handling**: 

1. The function, `erase_if()` , is used to erase a certain contour, which has an area less than 20, from the list of contours.

### Put Circle & Text

![Text_logic2](https://github.com/Joungbin-C/DLIP/blob/main/LAB/DLIP_LAB_Grayscale_Image_Segmentation_22200757/img/puttextdiagram.jpg)

1. Find the moment of the image which goes through opening morphology. The function, `moements()`, calculate the moments base on the location of pixels and intensity. In this project, that function is utilized to find the center point of the gear. The equation follows:

$$
(x,y) = ( \frac {Moments.m10} {Moments.m00}, \frac {Moments.m01} {Moments.m00})
$$

​		`m00` is the area

​		`m10` and `m01` are 1st spatial moments

2. With a same logic, find the center point of each contour of teeth. 
3. Subtract the center point of teeth to the center point of gear. It is the direction.

$$
Direction = (x_{gear}, y_{gear}) - (x_{teeth}, y_{teeth})
$$

4. Normalized direction with its length

$$
length = \sqrt{x_{direction}^2, y_{direction}^2}\\
Normalize(x_{direction}, y_{direction}) = (x_{direction}, y_{direction}) / length
$$

5. Find the text size with `getTextSize()` function. The center point of text box is a half of the text size. The reason of finding the text size is that `putText()` function places the text with the point of left-bottom point of the text box. 

$$
Center \ of \ text \ (x, y) = (textsize.height/2 , textsize.width/2)
$$

![Text_logic](https://github.com/Joungbin-C/DLIP/blob/main/LAB/DLIP_LAB_Grayscale_Image_Segmentation_22200757/img/textbox.jpg)

6. Find the point to place the text

$$
text\ position = gear\ center - normalized\ direction * (length-50)
\\
text\ position.y += textsize.height / 2;
\\
text\ position.x -= textsize.width / 2;
$$

### Show_Image()

The function, `Show_Image()`, has been created to see the image step by step. To see the next picture, any key must be pressed. 


# Result and Discussion

## 1. Final Result

![Gear1_static](https://github.com/Joungbin-C/DLIP/blob/main/LAB/DLIP_LAB_Grayscale_Image_Segmentation_22200757/img/Gear1_static.jpg)

![Gear1](https://github.com/Joungbin-C/DLIP/blob/main/LAB/DLIP_LAB_Grayscale_Image_Segmentation_22200757/img/Gear1.jpg)

![Gear2_static](https://github.com/Joungbin-C/DLIP/blob/main/LAB/DLIP_LAB_Grayscale_Image_Segmentation_22200757/img/Gear2_static.jpg)

![Gear2](https://github.com/Joungbin-C/DLIP/blob/main/LAB/DLIP_LAB_Grayscale_Image_Segmentation_22200757/img/Gear2.jpg)

![Gear3_static](https://github.com/Joungbin-C/DLIP/blob/main/LAB/DLIP_LAB_Grayscale_Image_Segmentation_22200757/img/Gear3_static.jpg)

![Gear3](https://github.com/Joungbin-C/DLIP/blob/main/LAB/DLIP_LAB_Grayscale_Image_Segmentation_22200757/img/Gear3.jpg)

![Gear4_static](https://github.com/Joungbin-C/DLIP/blob/main/LAB/DLIP_LAB_Grayscale_Image_Segmentation_22200757/img/Gear4_static.jpg)

![Gear4](https://github.com/Joungbin-C/DLIP/blob/main/LAB/DLIP_LAB_Grayscale_Image_Segmentation_22200757/img/Gear4.jpg)

Each Figure shows the statistical output and visual results. As images show, a teeth-only image is extracted from an original image. The area of each tooth is displayed. The defective teeth are colored red. Lastly, the defective teethes are circled on the original image. As the result shows defected teethes are well defined. 



## 2. Discussion

![Results](https://github.com/Joungbin-C/DLIP/blob/main/LAB/DLIP_LAB_Grayscale_Image_Segmentation_22200757/img/Results.jpg)

This algorithm successfully accomplishes the project goal, which is to classify defective teethes and normal teethes. As Table 1 shows, the total number of teethes, the average teeth area, the number of defective teethes, the diameter of the gear, and the quality classification. The broken or poorly constructed teethes are well detected, but the shape of teeth cannot be judged. It means that even if the teeth are poorly made, they will be detected as normal teeth if it is over the threshold value of area. Additionally, if different gears come in, the area threshold value has to be changed. In this experiment, the environment is fixed with the same size gears that are taken a picture at the same height. To utilize this algorithm for different sizes of gears, the method of finding the threshold is needed. Moreover, the morphology with a big-sized kernel takes a long time to process. The reason is that a bigger size kernel increases the computation complexity since it is a convolution calculation. Therefore, this algorithm is not efficient in detecting moving gears. However, it is efficient for this project environment, because it is detecting errors with pictures of gears. 



# Conclusion

The goal of this project was to detect defective teeth in gear images using image processing techniques. The algorithm successfully identified irregularities in the teeth based on contour analysis and area measurement. This method successfully detected all of the test images. It is to be used in real situations, adjusting thresholds and environments to increase accuracy and robustness. 
There are certain parts that should be developed if the environment or setting has been changed. Therefore, constructing the right algorithm for its environment is essential for vision engineering. 









---

# Appendix

```c++
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
void Calculate_Teeth_Properties(const vector<vector<Point>>& contours, const Point2f& gear_center, int& normal_teeth, int& defective_teeth, double& total_area, vector<double>& lengths,  double& total_length);
void Draw_Teeth_Contours(Mat& contourImg, Mat& img, const vector<vector<Point>>& contours, const Point2f& gear_center, const vector<double>& lengths, double avg_length);
void Printing_Teeth_Results(int normal_teeth, int defective_teeth, double total_area, float diameter);
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

// Calculate length, area, center of teeth, number of normal and defective teeth
void Calculate_Teeth_Properties(const vector<vector<Point>>& contours, const Point2f& gear_center, int& normal_teeth, int& defective_teeth, double& total_area, vector<double>& lengths, double& total_length) {
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
            defective_teeth++;
            lengths.push_back(length);
        }
    }
}

// Draw contour and put text next to it. Draw circle on defective teeth.
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
void Printing_Teeth_Results(int normal_teeth, int defective_teeth, double total_area, float diameter) {
    double total_teeth = normal_teeth + defective_teeth;
    double avg_area = total_teeth == 0 ? 0 : total_area / total_teeth;

    cout << "Total Teeth: " << total_teeth << endl;
    cout << "Avg Teeth Area: " << avg_area << endl;
    cout << "Normal Teeth: " << normal_teeth << endl;
    cout << "defective Teeth: " << defective_teeth << endl;
    cout << "Diameter: " << diameter << endl;

    if (defective_teeth != 0) {
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

    int normal_teeth = 0, defective_teeth = 0;
    double total_area = 0;
    vector<double> lengths = {0};
    double total_length = 0;
    Calculate_Teeth_Properties(contours, gear_center, normal_teeth, defective_teeth, total_area, lengths, total_length);

    double avg_length = lengths.empty() ? 0 : total_length / lengths.size();

    Mat contourImg = Mat::zeros(teeth_img.size(), CV_8UC3);
    Draw_Teeth_Contours(contourImg, img, contours, gear_center, lengths, avg_length);

    Show_Image(contourImg, "Detected Contours with Areas");
    Show_Image(img, "Defective Teeth Marked");

    Printing_Teeth_Results(normal_teeth, defective_teeth, total_area, diameter);
}
```
