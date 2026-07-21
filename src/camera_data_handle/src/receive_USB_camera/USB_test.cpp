#include "../../include/receive_USB_camera/receive_USB_camera.hpp"

using namespace std;
//
int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<receive_USB>());
    rclcpp::shutdown();
    cv::destroyAllWindows();
    return 0;
}