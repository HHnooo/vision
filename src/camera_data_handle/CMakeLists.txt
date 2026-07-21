cmake_minimum_required(VERSION 3.10)
project(camera_data_handle)

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

find_package(ament_cmake_auto REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(OpenCL REQUIRED)
find_package(rclcpp REQUIRED)
find_package(robot_interfaces REQUIRED)
ament_auto_find_build_dependencies()

#option(IF_USE_OPENVINO "Detection use openvino" ON)

#if(IF_USE_OPENVINO)
#    add_definitions(-DETECTION_OPENVINO_OPEN)
#    message(STATUS "OpenVINO support enabled-----------------------")
#else ()
#    message(STATUS "OpenVINO support unabled-----------------------!!!!!!!!!!")
#endif ()

add_definitions(-D USB_USE_YOLO)                #usb相机使用yolo检测目标
add_definitions(-D HIGHT_DEBUG)                 #不接受话题高度,是使用调试高度
#add_definitions(-D PID_PREDICT_OPEN)            #开启pid算法预测
#add_definitions(-D KALMAN_OPEN_DEBUG)           #开启卡尔曼跟踪目标调试
#add_definitions(-D KALMAN_OPEN)                 #开启卡尔曼跟踪目标
#add_definitions(-D RVIZ_DEBUG)                  #使用rviz显示出坐标变换
add_definitions(-D VIDEO_WRITE)                 #开启视频记录
#add_definitions(-D FPS_VISABLE_OPEN)            #开启fps显示
#add_definitions(-D YOLOV8_DETECTOR_OFF)         #关闭YOLOv8检测
#add_definitions(-D NO_IMAGE)                    #关闭图像显示
add_definitions(-D DETECTION_OPENVINO_OPEN)     #开启openvino加速
#add_definitions(-D RUN_THE_FUNCTION_BY_GPU)     #将周边函数放到GPU上
#add_definitions(-D KEY_POINT_TRACKING)          #开启关键点检测
#add_definitions(-D PREDICT_OPEN)                #开卡尔曼预测
#add_definitions(-D PUB_PRE_NOT_REAL)             #发送预测的目标点而非实际检测到的目标

add_definitions(-D PROJECT_PATH=${CMAKE_CURRENT_SOURCE_DIR})

function(add_and_install_executable target_name)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS LIBS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    add_executable(${target_name} ${ARG_SRCS})

    if (ARG_DEPS)
        ament_target_dependencies(${target_name} ${ARG_DEPS})
    endif ()

    if (ARG_LIBS)
        target_link_libraries(${target_name} ${ARG_LIBS})
    endif ()

    install(TARGETS ${target_name}
            DESTINATION lib/${PROJECT_NAME})
endfunction()

#add_and_install_executable(kalman_debug
#        SRCS    include/kalman_moving_target/test.cpp
#                include/kalman_moving_target/kalman.cpp
#        DEPS    rclcpp
#                OpenCV
#                tf2_eigen
#)

#add_and_install_executable(mytry
#        SRCS    src/try.cpp
#        LIBS    OpenCL
#)

add_and_install_executable(camera_debug
        SRCS    src/camera_debug/camera_debug.cpp
        DEPS    rclcpp
                OpenCV
                cv_bridge
                tf2
                tf2_geometry_msgs
)

add_and_install_executable(backup_debug
        SRCS    src/recieve_data/detector_backup.cpp
                src/backup_debug.cpp
                src/recieve_data/Yolov8Detector.cpp
        DEPS    rclcpp
                OpenCV
                cv_bridge
                tf2
                tf2_ros
                tf2_geometry_msgs
        LIBS    OpenCL
                openvino::runtime
)

#add_and_install_executable(predict_debug
#        SRCS    src/camera_debug/predict_debug.cpp
 #       DEPS    rclcpp
 #               OpenCV
#                cv_bridge
#                tf2
#                tf2_ros
 #               tf2_geometry_msgs
#)

#add_and_install_executable(receive_openmv_data
#        SRCS    src/receive_openmv_data/receive_openmv_data.cpp
#        DEPS    rclcpp
#                OpenCV

#)


add_and_install_executable(USB_camera
        SRCS    include/receive_USB_camera/Camera_driver.cpp
                src/receive_USB_camera/receive_USB_camera.cpp
                src/receive_USB_camera/USB_test.cpp
                src/recieve_data/Yolov8Detector.cpp
        DEPS    OpenCV
                tf2
                rclcpp
                tf2_eigen
                tf2_ros
                robot_interfaces
                sensor_msgs
        LIBS    OpenCL
                openvino::runtime
                yaml-cpp
)

add_and_install_executable(receive_data
        SRCS    src/recieve_data/receive_data.cpp
                src/recieve_data/Yolov8Detector.cpp
                src/recieve_data/camera_handle.cpp
                src/recieve_data/point_tracking.cpp
                src/recieve_data/detector_backup.cpp
                src/recieve_data/pid_predict.cpp
                #src/recieve_data/target_predict.cpp
                include/kalman_moving_target/kalman.cpp
        DEPS    rclcpp
                OpenCV
                cv_bridge
                tf2
                tf2_ros
                tf2_geometry_msgs
                yaml-cpp
                robot_interfaces
        LIBS    OpenCL
                openvino::runtime
)

add_and_install_executable(detect_publish
        SRCS    src/tf_publish/tf_publisher.cpp
                src/tf_publish/detect_publish.cpp
                src/recieve_data/Yolov8Detector.cpp
                src/receive_USB_camera/receive_USB_camera.cpp
                include/receive_USB_camera/Camera_driver.cpp
        DEPS    rclcpp
                OpenCV
                tf2
                tf2_ros
                tf2_eigen
                tf2_geometry_msgs
                robot_interfaces
                sensor_msgs
        LIBS    OpenCL
                openvino::runtime
                yaml-cpp
)

add_and_install_executable(calculate_publish
        SRCS    src/tf_publish/tf_publisher.cpp
                #src/receive_openmv_data/receive_openmv_data.cpp
                include/receive_USB_camera/Camera_driver.cpp
                src/tf_publish/calculate_publish.cpp
                src/receive_USB_camera/receive_USB_camera.cpp
                src/recieve_data/Yolov8Detector.cpp
        DEPS    rclcpp
                tf2
                OpenCV
                tf2_ros
                tf2_eigen
                tf2_geometry_msgs
                robot_interfaces
                sensor_msgs
        LIBS    OpenCL
                openvino::runtime
                yaml-cpp
)

if (BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    set(ament_cmake_copyright_FOUND TRUE)
    set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif ()

install(
        DIRECTORY config
        DESTINATION share/${PROJECT_NAME}
)

install(
        DIRECTORY launch
        DESTINATION share/${PROJECT_NAME}
)

ament_auto_package(
        INSTALL_TO_SHARE
        launch
)
