
#ifndef CAMERA_DATA_HANDLE_CAMERA_DEBUG_H
#define CAMERA_DATA_HANDLE_CAMERA_DEBUG_H

#include <rclcpp/rclcpp.hpp>
#include "sensor_msgs/msg/image.hpp"
#include "opencv4/opencv2/opencv.hpp"
#include <string>

namespace camera {

    class CameraDebug : public rclcpp::Node {
    public:
        explicit CameraDebug();

    private:
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
        rclcpp::TimerBase::SharedPtr timer_;

        cv::VideoCapture cap_;
        std::string video_path_;
        bool loop_;

        void publishFrame();
    };

}

#endif //CAMERA_DATA_HANDLE_CAMERA_DEBUG_H
