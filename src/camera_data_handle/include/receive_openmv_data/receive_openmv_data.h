#ifndef CAMERA_DATA_HANDLE_RECEIVE_OPENMV_DATA_H
#define CAMERA_DATA_HANDLE_RECEIVE_OPENMV_DATA_H

#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <cstdint>
#include <termios.h>
#include <fcntl.h>
#include <memory>
#include <unistd.h>
#include <sys/ioctl.h>
#include <opencv2/opencv.hpp>
#include "nlohmann/json.hpp"
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <Eigen/Eigen>
#include "cmath"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/transform_broadcaster.hpp"
#include "tf2/convert.hpp"
#include "../camera_params/camera_params.h"

///发布可视化坐标系的类
class TFDebug {
public:
    explicit TFDebug(std::string parent = "odom", std::string child = "default", double X = 0,
                     double Y = 0,
                     double Z = 0);

    void SetTransform(Eigen::Matrix3d transform);

    [[nodiscard]] rclcpp::Node::SharedPtr get_node() const {
        return node_;
    }

    void broadcast_transform();

private:
    std::string child_;
    std::string parent_;
    Eigen::Matrix3d transform_;
    double X_;
    double Y_;
    double Z_;
    rclcpp::Node::SharedPtr node_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};


/// 读取openmv的数据流
class ReceiveOpenMVData {
public:
    explicit ReceiveOpenMVData(std::string port = "/dev/ttyACM0", speed_t baudRate = B115200);

    std::string Receive_Openmv_Data();

private:
    const std::string port_;
    const speed_t baudRate_;

    static int openSerialPort(const std::string &port, speed_t baudRate);

    static std::string readLine(int fd);
};

///进行解算
class CalculateTarget {
public:
#ifdef RVIZ_DEBUG
    explicit CalculateTarget(std::shared_ptr<TFDebug> tf_debug,
                             const camera::CameraParams &params = camera::CameraParams{});
#endif
#ifndef RVIZ_DEBUG
    explicit CalculateTarget(const camera::CameraParams &params = camera::CameraParams{});
#endif
    cv::Point2d Handle_Openmv_Data(const geometry_msgs::msg::TransformStamped &pose);

private:
    // 创建接受信息的ReceiveOpenMVData指针
    std::shared_ptr<ReceiveOpenMVData> receive_openmv_data_;

    // 从YAML加载的相机参数
    camera::CameraParams params_;

    // 将受到的字符串解码
    cv::Point2d Decode_Openmv_Data(std::string &input_str);

    // 将得到的像素坐标解算(主要算法)
    cv::Point2d Transform_Image_TO_Real(cv::Point2d &image_point, geometry_msgs::msg::TransformStamped pose);

#ifdef RVIZ_DEBUG
    std::shared_ptr<TFDebug> tf_debug_ptr_;
#endif

};


#endif //CAMERA_DATA_HANDLE_RECEIVE_OPENMV_DATA_H
