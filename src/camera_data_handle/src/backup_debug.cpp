#include "../include/camera_data_handle/detector_backup.h"

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<camera::DetectorBackup>());
    rclcpp::shutdown();
    cv::destroyAllWindows();
    return 0;
}