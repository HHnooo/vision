#ifndef CAMERA_DATA_HANDLE_PREDICT_DEBUG_H
#define CAMERA_DATA_HANDLE_PREDICT_DEBUG_H

#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <tf2/transform_datatypes.hpp>
#include <tf2_ros/transform_broadcaster.hpp>
#include "tf2/exceptions.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"

namespace predict_debug {
    class Tf2Listener : public rclcpp::Node {
    public:
        Tf2Listener();

    private:

        void on_timer();

        bool if_the_service_ready_{false};

        rclcpp::TimerBase::SharedPtr timer_{nullptr}; //时间
        std::unique_ptr<tf2_ros::Buffer> tf_buffer_{nullptr};
        std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};



    };
}


#endif //CAMERA_DATA_HANDLE_PREDICT_DEBUG_H
