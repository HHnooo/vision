#ifndef CAMERA_HANDLE_H
#define CAMERA_HANDLE_H

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "opencv4/opencv2/opencv.hpp"
#include "cv_bridge/cv_bridge.h"
#include "std_msgs/msg/float64_multi_array.hpp"
#include <robot_interfaces/msg/measure.hpp>
#include <robot_interfaces/msg/kalman_output.hpp>
#include "opencv2/videoio.hpp"
#include <std_msgs/msg/bool.hpp>

#include "../PID_predect/pid_predict.hpp"
#include "../Yolov8Detector/Yolov8Detector.h"
#include "../kalman/kalmanbox.h"
#include "target_predict.h"
#include "../point_tracking/point_tracking.h"
#include "robot_interfaces/msg/image_location.hpp"
#include "../kalman_moving_target/kalman.hpp"


namespace camera {
    using KalmanOutput = robot_interfaces::msg::KalmanOutput;
    using Measure = robot_interfaces::msg::Measure;

    static kalman::KalmanInput transform_to_karman_input(const Yolov8::Detection &detection);

    static Eigen::VectorXd transform_to_eigen_vector(const kalman::KalmanInput &input);

    class ReceiveData : public rclcpp::Node {
    public:
        explicit ReceiveData();

        ~ReceiveData() override;

    private:
        void imageCallback(sensor_msgs::msg::Image::ConstSharedPtr msg);

        rclcpp::Publisher<robot_interfaces::msg::ImageLocation>::SharedPtr position_pub_;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
        rclcpp::Subscription<geometry_msgs::msg::TransformStamped>::SharedPtr pose_sub_;
        rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr twist_sub_; //接受速度的
        rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr camera_switch_sub_;
        rclcpp::TimerBase::SharedPtr timer_;
        std::vector<std::string> output_names_;
        std::vector<std::string> class_names_;
        geometry_msgs::msg::TransformStamped pose_;
        geometry_msgs::msg::Twist twist_;
        double confidence_threshold_{0.0};
        double nms_threshold_{0.0};
        double fps_time_sum_{0.0};
        int tracking_num_{0};
        int fps_sum_{0};
        bool if_do_tracking_{false}; //如果检测或者跟踪都失效时设置为false
        bool has_received_{false};
        bool if_camera_enabled_{true}; //默认使能检测
        std::string model_path_;
        cv::TickMeter fps_timer_;
        cv::dnn::Net net_;
#ifdef PID_PREDICT_OPEN
        int max_running{0};
        cv::Point2f predict_result_;
        V_Predict v_predict_;
        TargetSpeedEstimator estimator;
        rclcpp::Publisher<Measure>::SharedPtr measure_pub_;
#endif
#ifdef KALMAN_OPEN_DEBUG
        rclcpp::Publisher<Measure>::SharedPtr measure_pub_;
        rclcpp::Publisher<KalmanOutput>::SharedPtr kalman_pub_;
#endif
#ifdef KALMAN_OPEN
        int kalman_step_{0};
        std::shared_ptr<kalman::Kalman> kf_;
        kalman::KalmanOutput kf_result_;
        std::atomic<bool> is_meas_unsuccessful_{true};
        Eigen::VectorXd current_meas_;
#endif
#ifndef KEY_POINT_TRACKING
        Yolov8::YOLOv8Detector detector_;
#endif
#ifdef KEY_POINT_TRACKING
        Yolov8::YOLOv8Tracker detector_;
#endif
#ifdef  VIDEO_WRITE
        int frameWidth_ = 640;
        int frameHeight_ = 480;
        std::string video_name_;
        cv::VideoWriter writer_;
#endif
#ifdef PREDICT_OPEN
        camera::KalmanBoxTracker *tracker_{nullptr}; //卡尔曼滤波的指针
#endif
    };
}


#endif //CAMERA_HANDLE_H