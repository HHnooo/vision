#include "../../include/camera_data_handle/camera_handle.h"

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<camera::ReceiveData>());
    rclcpp::shutdown();
    cv::destroyAllWindows();
    return 0;
}
