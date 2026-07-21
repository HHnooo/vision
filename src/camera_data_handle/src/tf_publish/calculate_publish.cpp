#include "../../include/tf_publish/tf_publish.h"

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
#ifdef RVIZ_DEBUG
    auto tf_broadcaster = std::make_shared<TFDebug>();
    auto calculate = std::make_shared<camera::Calculate_Publisher>(tf_broadcaster);
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(calculate);
    executor.add_node(tf_broadcaster->get_node());
    executor.spin();
#endif

#ifndef RVIZ_DEBUG
    rclcpp::spin(std::make_shared<camera::Calculate_Publisher>());
    //rclcpp::spin(std::make_shared<camera::Detect_Publisher>());
#endif

    rclcpp::shutdown();
    return 0;
}
