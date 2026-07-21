#ifndef TF_PUBLISH_H
#define TF_PUBLISH_H

#define THE_TRANSFORM_USE_PREDICT
#include "../receive_USB_camera/receive_USB_camera.hpp"

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>   //高度的消息类型
#include <std_msgs/msg/bool.hpp>

#include "tf2_ros/transform_broadcaster.h"
#include "../receive_openmv_data/receive_openmv_data.h"
#include "robot_interfaces/msg/image_location.hpp"
#include "../receive_USB_camera/receive_USB_camera.hpp"
#include <opencv4/opencv2/opencv.hpp>
#include "../Yolov8Detector/Yolov8Detector.h"

namespace camera {
    enum Target {
        H = 0,
        TENT = 1,
        CAR = 2,
        BRIDGE = 3,
        PILLBOX = 4,
        TANK = 5,
        RED_CROSS = 6,
        UNKNOW = 7,
        INITIALIZER = 8,
    };

    struct DetectVector {
        cv::Point point;
        int ClassId;
    };

    ///基类--------------------------------------------------------------------------------
    class TF_Publisher_Base {
    public:
        virtual ~TF_Publisher_Base() = default;

        TF_Publisher_Base();

    protected:
        virtual void publish_transform();

        static void initialize_transform(robot_interfaces::msg::ImageLocation &msg_loader);

        //隔断时间
        rclcpp::TimerBase::SharedPtr send_timer_;

        //发布信息的容器
        robot_interfaces::msg::ImageLocation pub_pos_;

        //发送的信息(这个是具体地识别信息)
        geometry_msgs::msg::TransformStamped transform_;

        bool transform_initialized;

        //测试用的高度
        double height_;

        //静态变量(看是否能将d435i的数据用作猜测随机靶数据)
        // static uint8_t use_this_or_camera_pub_msg_;

        // static float guess_x;
        // static float guess_y;
    };

    ///继承的目标检测的类-------------------------------------------------------------------------------------
    class Detect_Publisher : public TF_Publisher_Base, public rclcpp::Node {
    public:
        Detect_Publisher();

        ~Detect_Publisher() override = default;

    private:
        void publish_transform() override;

        void position_callback(const robot_interfaces::msg::ImageLocation::SharedPtr msg);

#ifndef HIGHT_DEBUG

        //接受高度
        rclcpp::Subscription<geometry_msgs::msg::TransformStamped>::SharedPtr sub_height_;

        void HeightCallback(geometry_msgs::msg::TransformStamped::SharedPtr msg);

#endif

        //自定义接口信息发送
        rclcpp::Publisher<robot_interfaces::msg::ImageLocation>::SharedPtr position_pub_;

        //接受点的信息
        rclcpp::Subscription<robot_interfaces::msg::ImageLocation>::SharedPtr position_subscription;

        //更新计数器（有时候丢失目标，能够延时）
        long tf2_reflash_;

        //最大更新次数
        int tf2_reflash_num_;
    };

    ///继承的猜测openmv的类----------------------------------------------------------------------------------------------------------
    class Calculate_Publisher : public TF_Publisher_Base, public rclcpp::Node {
    public:
#ifdef RVIZ_DEBUG
        explicit Calculate_Publisher(std::shared_ptr<TFDebug> tf_debug);
#endif
#ifndef RVIZ_DEBUG
        explicit Calculate_Publisher();
#endif

        ~Calculate_Publisher() override = default;

    private:
        void publish_transform() override;

        rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr camera_switch_sub_;

        bool if_can_start_USB_{false};

#ifndef HIGHT_DEBUG
        //接受高度
        rclcpp::Subscription<geometry_msgs::msg::TransformStamped>::SharedPtr sub_pose_;

        void PoseCallback(geometry_msgs::msg::TransformStamped::SharedPtr msg);
#endif

        //储存位姿
        geometry_msgs::msg::TransformStamped_<std::allocator<void> > *pose_;

        //自定义接口信息发送
        rclcpp::Publisher<robot_interfaces::msg::ImageLocation>::SharedPtr position_pub_;

        //std::shared_ptr<CalculateTarget> calculate_target_class_;
        usb_camera::USBFactor usb_factor_;

        //openmv的信息
        geometry_msgs::msg::TransformStamped transform_openmv_;

#ifdef RVIZ_DEBUG
        //传递的指针
        std::shared_ptr<TFDebug> tf_debug_;
#endif
    };
}


#endif //TF_PUBLISH_H