#include "../../include/camera_data_handle/camera_debug.h"

#include "rclcpp/rclcpp.hpp"

using namespace cv;
using namespace std;

camera::CameraDebug::CameraDebug() : Node("camera_debug") {

    // 通过 ROS 参数指定视频路径，默认为 vidios 目录下的第一个视频
    this->declare_parameter<string>("video_path", "../vidios/vidio0001.mp4");
    this->declare_parameter<bool>("loop", true);

    video_path_ = this->get_parameter("video_path").as_string();
    loop_ = this->get_parameter("loop").as_bool();

    RCLCPP_INFO(this->get_logger(), "打开视频文件: %s", video_path_.c_str());
    RCLCPP_INFO(this->get_logger(), "循环播放: %s", loop_ ? "是" : "否");

    cap_.open(video_path_);
    if (!cap_.isOpened()) {
        RCLCPP_ERROR(this->get_logger(), "无法打开视频文件: %s", video_path_.c_str());
        RCLCPP_ERROR(this->get_logger(), "请确认文件路径正确，或通过参数指定: --ros-args -p video_path:=<路径>");
        rclcpp::shutdown();
        return;
    }

    RCLCPP_INFO(this->get_logger(), "视频 %.0f×%.0f, %.0ffps, %d帧",
                cap_.get(CAP_PROP_FRAME_WIDTH),
                cap_.get(CAP_PROP_FRAME_HEIGHT),
                cap_.get(CAP_PROP_FPS),
                (int)cap_.get(CAP_PROP_FRAME_COUNT));

    pub_ = this->create_publisher<sensor_msgs::msg::Image>(
        "/camera/camera/color/image_raw", 10);

    // 按视频原帧率发布
    double video_fps = cap_.get(CAP_PROP_FPS);
    int interval_ms = video_fps > 0 ? (int)(1000.0 / video_fps) : 33;

    timer_ = this->create_wall_timer(
        chrono::milliseconds(interval_ms),
        bind(&CameraDebug::publishFrame, this));
}

void camera::CameraDebug::publishFrame() {
    Mat frame;
    cap_ >> frame;

    // 播放完毕：根据 loop_ 决定循环或退出
    if (frame.empty()) {
        if (loop_) {
            cap_.set(CAP_PROP_POS_FRAMES, 0);
            RCLCPP_INFO(this->get_logger(), "视频循环播放");
            return;
        } else {
            RCLCPP_INFO(this->get_logger(), "视频播放完毕，退出");
            rclcpp::shutdown();
            return;
        }
    }

    sensor_msgs::msg::Image img_msg;

    cvtColor(frame, frame, COLOR_BGR2RGB);

    img_msg.encoding = "rgb8";
    img_msg.header.frame_id = "camera";
    img_msg.width = frame.cols;
    img_msg.height = frame.rows;
    img_msg.step = frame.step;

    size_t size = frame.step * frame.rows;
    img_msg.data.resize(size);
    memcpy(&img_msg.data[0], frame.data, size);

    img_msg.header.stamp = rclcpp::Clock().now();

    pub_->publish(img_msg);
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = make_shared<camera::CameraDebug>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    destroyAllWindows();
    return 0;
}
