#include <yaml-cpp/node/node.h>

#include "../../include/tf_publish/tf_publish.h"

#define MAX_STEP 0.8

#ifdef DETECTION_OPENVINO_OPEN
#warning "DETECTION_OPENVINO_OPEN is defined"
#else
#warning "DETECTION_OPENVINO_OPEN is NOT defined"
#endif

uint8_t use_this_or_camera_pub_msg_{0};
float guess_x{0};
float guess_y{0};

camera::TF_Publisher_Base::TF_Publisher_Base() : transform_initialized(false), height_(0.8) {
    use_this_or_camera_pub_msg_ = 0;
    guess_x = 0;
    guess_y = 0;
}

void camera::TF_Publisher_Base::publish_transform() {
}

void camera::TF_Publisher_Base::initialize_transform(robot_interfaces::msg::ImageLocation &msg_loader) {
    msg_loader.image_x = 0;
    msg_loader.image_y = 0;
    msg_loader.id = INITIALIZER;
}

///继承的目标检测的类
///---------------------------------------------------------------------------------------------------------------------
camera::Detect_Publisher::Detect_Publisher() : Node("Detect_Publisher"), tf2_reflash_(0), tf2_reflash_num_(0) {
    RCLCPP_INFO(this->get_logger(), "TF_Publisher");

    // this->declare_parameter("height", 1.5);
    // this->declare_parameter("tf2_reflash_num", 1);
    //
    // this->get_parameter("tf2_reflash_num", tf2_reflash_num_);
    // this->get_parameter("height", height_);

    position_pub_ = this->create_publisher<robot_interfaces::msg::ImageLocation>("/robot/image_location", 10);

#ifndef HIGHT_DEBUG
    sub_height_ = this->create_subscription<geometry_msgs::msg::TransformStamped>(
        "/robot/current_pose",
        10,
        std::bind(&Detect_Publisher::HeightCallback, this, std::placeholders::_1)
    );
#endif

    position_subscription = this->create_subscription<robot_interfaces::msg::ImageLocation>(
        "/camera/target/position",
        10,
        std::bind(&Detect_Publisher::position_callback, this, std::placeholders::_1)
    );

    send_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(10),
        std::bind(&Detect_Publisher::publish_transform, this));
}

#ifndef HIGHT_DEBUG
void camera::Detect_Publisher::HeightCallback(geometry_msgs::msg::TransformStamped::SharedPtr msg) {
    ///这里加了个0.39-------------------------------------------------------------------------
    double height = msg.get()->transform.translation.z + 0.1;
    height_ = height;
}

#endif

void camera::Detect_Publisher::position_callback(const robot_interfaces::msg::ImageLocation::SharedPtr msg) {
    //std::cout<<tf2_reflash_num<<std::endl;
    if (msg.get()->id == UNKNOW) //处理为空的情况
    {
        tf2_reflash_++;
        if (tf2_reflash_ >= tf2_reflash_num_) {
            pub_pos_.image_x = 0;
            pub_pos_.image_y = 0;
            pub_pos_.id = UNKNOW;
            tf2_reflash_ = 0;
        }
        return;
    }

    tf2_reflash_ = 0; // 更新

    float x = msg->image_x;
    float y = msg->image_y;
    uint8_t status = msg->id;
    //std::cout << x << " " << y << " " << status << std::endl;

    // if (status == RED_CROSS) {
    //     if (use_this_or_camera_pub_msg_ >= 10) {
    //         guess_x = x;
    //         guess_y = y;
    //         use_this_or_camera_pub_msg_ = 10;
    //     }
    // }

#ifdef THE_TRANSFORM_USE_PREDICT
    //防止偏移位置过大
    pub_pos_.image_x = x * 42 * height_ / 20700;
    if (pub_pos_.image_x > MAX_STEP) {
        pub_pos_.image_x = MAX_STEP;
    }
    if (pub_pos_.image_x < -MAX_STEP) {
        pub_pos_.image_x = -MAX_STEP;
    }
    pub_pos_.image_y = y * 42 * height_ / 20700;
    if (pub_pos_.image_y > MAX_STEP) {
        pub_pos_.image_y = MAX_STEP;
    }
    if (pub_pos_.image_y < -MAX_STEP) {
        pub_pos_.image_y = -MAX_STEP;
    }
    pub_pos_.id = status;

    RCLCPP_INFO(this->get_logger(), "x: %f, y: %f, status: %i", pub_pos_.image_x,
                pub_pos_.image_y, pub_pos_.id);
#endif

#ifdef THE_TRANSFORM_USE_ACCELERATE
    constexpr double param = 0;

    pub_pos_.image_x = 0;
    pub_pos_.image_y = 0;
    pub_pos_.status = UNKNOW;

#endif
}

void camera::Detect_Publisher::publish_transform() {
    if (!transform_initialized) {
        initialize_transform(pub_pos_);
        transform_initialized = true;
    }
    position_pub_->publish(pub_pos_);
}

///继承的猜测openmv的类
///---------------------------------------------------------------------------------------------------------------------
#ifdef RVIZ_DEBUG
camera::Calculate_Publisher::Calculate_Publisher(std::shared_ptr<TFDebug> tf_debug)
    : Node("Calculate_Publisher"),
      usb_factor_(camera::CameraParams::load(CAMERA_YAML_PATH, "usb_camera")),
      tf_debug_(std::move(tf_debug))
#endif
#ifndef  RVIZ_DEBUG
camera::Calculate_Publisher::Calculate_Publisher()
    : Node("Calculate_Publisher"),
      usb_factor_(camera::CameraParams::load(CAMERA_YAML_PATH, "usb_camera"))
#endif
{
    RCLCPP_INFO(this->get_logger(), "TF_Publisher");
#ifdef RVIZ_DEBUG
    auto openmv_params = camera::CameraParams::load(CAMERA_YAML_PATH, "openmv_camera");
    calculate_target_class_ = std::make_shared<CalculateTarget>(tf_debug_, openmv_params);
#endif
#ifndef RVIZ_DEBUG
    //calculate_target_class_ = std::make_shared<CalculateTarget>();
#endif

    position_pub_ = this->create_publisher<robot_interfaces::msg::ImageLocation>("/robot/usb_camera", 10);
    camera_switch_sub_ = this->create_subscription<std_msgs::msg::Bool>(
        "/camera/choose",
        10,
        [this](const std_msgs::msg::Bool::ConstSharedPtr msg) {
            if_can_start_USB_ = msg->data;
        }
    );

#ifndef HIGHT_DEBUG
    sub_pose_ = this->create_subscription<geometry_msgs::msg::TransformStamped>(
        "/robot/current_pose",
        10,
        std::bind(&Calculate_Publisher::PoseCallback, this, std::placeholders::_1)
    );
#endif

    ///世界系：x向前，y向左，z向上
    ///无人机系：x向前，y向左，z向上
    ///相机系：x向右，y向下，z向前
    pose_ = new geometry_msgs::msg::TransformStamped_<std::allocator<void> >();

#ifdef HIGHT_DEBUG
    pose_->transform.rotation.w = 1;
    pose_->transform.rotation.x = 0;
    pose_->transform.rotation.y = 0;
    pose_->transform.rotation.z = 0;
    pose_->transform.translation.x = 0;
    pose_->transform.translation.y = 0;
    pose_->transform.translation.z = 0.85;   // 0.85 + height_offset(0.15) = 1.0m
#endif
    send_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(10),
        std::bind(&Calculate_Publisher::publish_transform, this));
}

#ifndef HIGHT_DEBUG
void camera::Calculate_Publisher::PoseCallback(geometry_msgs::msg::TransformStamped::SharedPtr msg) {
    pose_->transform = msg->transform;
    double w = pose_->transform.rotation.w;
    double x = pose_->transform.rotation.x;
    double y = pose_->transform.rotation.y;
    double z = pose_->transform.rotation.z;

    double height = pose_->transform.translation.z + 0.15;
    height_ = height;

    // pose_->transform.rotation.w = -x;
    // pose_->transform.rotation.x = w;
    // pose_->transform.rotation.y = -z;
    // pose_->transform.rotation.z = y;

    RCLCPP_INFO(this->get_logger(), "rotation: w:%f x:%f y:%f z:%f", pose_->transform.rotation.w,
                pose_->transform.rotation.x, pose_->transform.rotation.y, pose_->transform.rotation.z);
    RCLCPP_INFO(this->get_logger(), "translation: x:%f y:%f z:%f", pose_->transform.translation.x,
                pose_->transform.translation.y, pose_->transform.translation.z);
}
#endif

void camera::Calculate_Publisher::publish_transform() {
    if (!transform_initialized) {
        initialize_transform(pub_pos_);
        transform_initialized = true;
    }
    //受到false的时候不执行神经网络
    if (if_can_start_USB_) {
        RCLCPP_INFO(this->get_logger(), "Starting USB");
        return;
    }

    //cv::Point2d guess_point_ = calculate_target_class_.get()->Handle_Openmv_Data(*pose_);
    //std::vector<DetectVector> _output_points = usb_factor_.Receive_Keypoint();
    auto _output_points = usb_factor_.Receive_Keypoint();
    if (_output_points.empty()) {
        pub_pos_.image_x = 0;
        pub_pos_.image_y = 0;
        pub_pos_.id = UNKNOW;
        //std::cout << pub_pos_.image_x << " " << pub_pos_.image_y << std::endl;
        RCLCPP_INFO(this->get_logger(), "No Keypoints available");
        position_pub_->publish(pub_pos_);
    }

    for (auto &_output_point: _output_points) {
        cv::Point2d output;
        output.x = _output_point.point.x;
        output.y = _output_point.point.y;

        cv::Point2d guess_point_ = usb_factor_.Transform_Image_TO_Real(output, *pose_);

        pub_pos_.image_x = guess_point_.x;
        pub_pos_.image_y = guess_point_.y;

        // 新模型 class id 与 Target 枚举对齐: 0=H 1=tent 2=car 3=bridge 4=pillbox 5=tank 6=Red_cross
        pub_pos_.id = _output_point.ClassId;
        std::cout << usb_camera::ClassNames[_output_point.ClassId] << ": "
                  << guess_point_.x << " " << guess_point_.y << std::endl;
        position_pub_->publish(pub_pos_);
    }
}