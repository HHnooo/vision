#ifndef CAMERA_DATA_HANDLE_DETECTOR_BACKUP_H
#define CAMERA_DATA_HANDLE_DETECTOR_BACKUP_H

#include "../Yolov8Detector/Yolov8Detector.h"
#include "rclcpp/rclcpp.hpp"
#include <opencv4/opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

namespace camera {
    class DetectorBackup : public rclcpp::Node {
    public:
        explicit DetectorBackup();

    private:
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;

        cv::Mat image_;
        cv::Mat src_, hsv_, mask_;
        int hmin_ = 0, smin_ = 0, vmin_ = 0;
        int hmax_ = 179, smax_ = 255, vmax_ = 255;

        void imageCallback(sensor_msgs::msg::Image::ConstSharedPtr msg);
        static void on_trackbar(int, void* userdata) ;
    };

    // class DetectorBackupFactory {
    // public:
    //     explicit DetectorBackupFactory();
    //
    //     Yolov8::Detection detect(cv::Mat image);
    //
    //     ~DetectorBackupFactory();
    //
    // private:
    //     std::string class_name_;
    // };
}

#endif //CAMERA_DATA_HANDLE_DETECTOR_BACKUP_H
