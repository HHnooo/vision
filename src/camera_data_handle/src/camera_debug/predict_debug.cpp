#include "../../include/camera_data_handle/predict_debug.h"

predict_debug::Tf2Listener::Tf2Listener() : Node("predict_debug") {
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());

    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    timer_ = this->create_wall_timer(std::chrono::seconds(1), std::bind(&Tf2Listener::on_timer, this));
}


void predict_debug::Tf2Listener::on_timer() {
    std::string fromFrameRel = "camera_color_optical_frame";
    std::string toFrameRel = "target_position";
    geometry_msgs::msg::TransformStamped t;
    t = tf_buffer_->lookupTransform(
        toFrameRel, fromFrameRel,
        tf2::TimePointZero);
    RCLCPP_INFO(this->get_logger(), "Received frame from %f", t.transform.translation.x);
    RCLCPP_INFO(this->get_logger(), "Received frame from %f", t.transform.translation.y);
    RCLCPP_INFO(this->get_logger(), "Received frame from %f", t.transform.translation.z);
    RCLCPP_INFO(this->get_logger(), "---------------------------------------------------------");
}


int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<predict_debug::Tf2Listener>());
    rclcpp::shutdown();
}
