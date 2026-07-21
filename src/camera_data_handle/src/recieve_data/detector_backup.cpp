#include "../../include/camera_data_handle/detector_backup.h"
#include <opencv4/opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

using namespace cv;

camera::DetectorBackup::DetectorBackup() : Node("backup") {
    namedWindow("image", WINDOW_AUTOSIZE);
    namedWindow("hsv_image", WINDOW_AUTOSIZE);
    namedWindow("mask_image", WINDOW_AUTOSIZE);
    namedWindow("Mask", WINDOW_AUTOSIZE);

    // createTrackbar("Hue Min", "Mask", &hmin_, 179, on_trackbar, this);
    // createTrackbar("Hue Max", "Mask", &hmax_, 179, on_trackbar, this);
    createTrackbar("Hue Min", "Mask", &hmin_, 180, on_trackbar, this);
    createTrackbar("Hue Max", "Mask", &hmax_, 180, on_trackbar, this);
    createTrackbar("Sat Min", "Mask", &smin_, 255, on_trackbar, this);
    createTrackbar("Sat Max", "Mask", &smax_, 255, on_trackbar, this);
    createTrackbar("Val Min", "Mask", &vmin_, 255, on_trackbar, this);
    createTrackbar("Val Max", "Mask", &vmax_, 255, on_trackbar, this);

    on_trackbar(1, this);

    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera/camera/color/image_raw",
        10,
        std::bind(&camera::DetectorBackup::imageCallback, this, std::placeholders::_1)
    );
}

void camera::DetectorBackup::imageCallback(sensor_msgs::msg::Image::ConstSharedPtr msg) {
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
    cv::Mat temp_image = cv_ptr->image;
    cv::cvtColor(temp_image, image_, cv::COLOR_BGR2RGB);
    //cv::cvtColor(image_, hsv_, cv::COLOR_BGR2HSV);
    cv::cvtColor(image_, hsv_, cv::COLOR_BGR2HSV);

    Scalar lower(hmin_, smin_, vmin_);
    Scalar upper(hmax_, smax_, vmax_);
    //inRange(hsv_, lower, upper, mask_);
    inRange(hsv_, lower, upper, mask_);

    imshow("image", image_);
    imshow("hsv_image", hsv_);
    if (!mask_.empty()) {
        cv::Mat new_image ;
        Mat kernal=getStructuringElement(MORPH_RECT, Size(3, 3),Point(-1,-1));
        erode(mask_, new_image, kernal);
        imshow("mask_image", new_image);
    }

    cv::waitKey(20);
}

void camera::DetectorBackup::on_trackbar(int, void *) {
}
