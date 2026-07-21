#include "../../include/receive_USB_camera/receive_USB_camera.hpp"

#define STR(s) #s
#define MACRO_TO_STR(s) STR(s)

using namespace std;
using namespace cv;


receive_USB::receive_USB() : Node("USB_pub") {
    // 从YAML读取USB相机参数
    auto params = camera::CameraParams::load(CAMERA_YAML_PATH, "usb_camera");

    color_lower_ = Scalar(0, 0, 0);
    color_upper_ = Scalar(180, 255, 255);
    usb_camera_ = std::make_shared<usb_camera::USBCamera>(params.device_index);
    usb_camera_->OpenCameraDevice();
    usb_camera_->SetExposure(250);
    usb_camera_->SetResolution(camera::IMAGE_WIDTH, camera::IMAGE_HEIGHT);
    usb_camera_->SetFPS(30);
    pub_ = this->create_publisher<sensor_msgs::msg::Image>("/camera/camera/color/image_raw", 10);
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(30),
        std::bind(&receive_USB::callback, this)
    );
}

void receive_USB::callback() {
    sensor_msgs::msg::Image img_msg;
    cv::Mat frame;

    frame = usb_camera_->GetFrame();
    if (frame.empty()) return;
    image_ = frame.clone();
    // Receive_Keypoint();  // 调试用，正常运行时关闭
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    img_msg.encoding = "rgb8";
    img_msg.header.frame_id = "camera";
    img_msg.width = frame.cols;
    img_msg.height = frame.rows;
    img_msg.step = frame.step;

    size_t size = frame.step * frame.rows;
    img_msg.data.resize(size);
    memcpy(&img_msg.data[0], frame.data, size);

    img_msg.header.stamp = rclcpp::Clock().now();

    pub_->publish(img_msg);
}

cv::Point2f receive_USB::Receive_Keypoint() {
    Mat hsv, mask;
    image_ = usb_camera_->GetFrame();
    cvtColor(image_, hsv, COLOR_BGR2HSV);
    inRange(hsv, color_lower_, color_upper_, mask);

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    erode(mask, mask, kernel);
    dilate(mask, mask, kernel);

    vector<vector<Point> > contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<Point> max_contours;
    double max_contour_score = 200;

    for (size_t i = 0; i < contours.size(); i++) {
        if (contourArea(contours[i]) > max_contour_score) {
            max_contour_score = contours[i].size();
            max_contours = contours[i];
        }

        Rect bounding_rect = boundingRect(max_contours);

        rectangle(image_, bounding_rect, Scalar(0, 255, 0), 2);

        putText(image_, "Blue Object", Point(bounding_rect.x, bounding_rect.y - 10),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
    }

    Point2f output;
    output.x = (max_contours.at(0).x + max_contours.at(2).x) / 2;
    output.y = (max_contours.at(0).y + max_contours.at(1).y) / 2;
    imshow("11111", image_);
    waitKey(30);
    return output;
}

///--------------------------------------------------------------------------------------------------------------
usb_camera::USBFactor::USBFactor(const camera::CameraParams &params)
    : params_(params)
#ifdef USB_USE_YOLO
    , detector_(std::string(MACRO_TO_STR(PROJECT_PATH)) + "/model/nuc_best.xml", ClassNames, 640, 0.4), times(0)
#endif
{
    color_lower_ = Scalar(168, 68, 82);
    color_upper_ = Scalar(180, 255, 210);
    color_lower_ = Scalar(20, 0, 0);
    color_upper_ = Scalar(160, 255, 255);
    usb_camera_ = std::make_shared<usb_camera::USBCamera>(params_.device_index);
    usb_camera_->OpenCameraDevice();
    usb_camera_->SetExposure(100);
    usb_camera_->SetResolution(camera::IMAGE_WIDTH, camera::IMAGE_HEIGHT);
    usb_camera_->SetFPS(15);
}
#ifndef USB_USE_YOLO
cv::Point2d usb_camera::USBFactor::Receive_Keypoint() {
    cv::Mat hsv, mask, mask_b, grey;
    image_ = usb_camera_->GetFrame();
    cvtColor(image_, hsv, COLOR_BGR2HSV);
    cvtColor(image_, grey, COLOR_BGR2GRAY);

    inRange(hsv, color_lower_, color_upper_, mask);
    inRange(hsv, Scalar(0, 140, 100), Scalar(180, 230, 255), mask_b);

    bitwise_not(mask, mask);
    bitwise_and(mask, mask_b, mask);

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    erode(mask, mask, kernel);
    dilate(mask, mask, kernel);
    //erode(image_, image_, kernel);


    vector<vector<Point> > contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<Point> max_contours;
    double max_contour_score = 100;

    for (const auto &contour: contours) {
        Rect bounding_rect = boundingRect(contour);
        if (bounding_rect.width * bounding_rect.height > max_contour_score) {
            max_contour_score = bounding_rect.width * bounding_rect.height;
            max_contours = contour;
        }
    }

    Rect bounding_rect = boundingRect(max_contours);

    double rect_w = bounding_rect.width;
    double rect_h = bounding_rect.height;

    if (rect_h / rect_w < 1 / 2) {
        return {camera::NULL_ERROR, camera::NULL_ERROR};
    }
    /// -----------------------------------------------------------------------------------------------------------------

    vector<Point> output_points;
    vector<Point> coners;
    Point max_point;
    Point min_point;
    double max_way = 0;
    double min_way = 641;
    Point2f centor(bounding_rect.x + bounding_rect.width / 2, bounding_rect.y + bounding_rect.height / 2);

    goodFeaturesToTrack(mask, coners, 50, 0.7, 2);
    cout << coners.size() << endl;

    if (coners.size() > 30) {
        imshow("22222", mask);
        imshow("11111", image_);
        waitKey(30);
        return {camera::NULL_ERROR, camera::NULL_ERROR};
    }

    // if ( max_contours.size() > 64) {
    //
    //     approxPolyN(max_contours, output_points, 8);
    //
    //     for (auto &point: output_points) {
    //         double way =
    //                 sqrt((point.x - centor.x) * (point.x - centor.x) + (point.y - centor.y) * (point.y - centor.y));
    //         if (way < min_way) {
    //             min_way = way;
    //             min_point = point;
    //         }
    //         if (way > max_way) {
    //             max_way = way;
    //             max_point = point;
    //         }
    //     }
    //     cout<<coners.size()<<endl;
    //     if (min_way / max_way < 1 / 2) {
    //         return {camera::NULL_ERROR,camera::NULL_ERROR};
    //     }
    // } else {
    //     imshow("22222", mask);
    //     imshow("11111", image_);
    //     waitKey(30);
    //     return {camera::NULL_ERROR,camera::NULL_ERROR};
    // }
    ///------------------------------------------------------------------------------------------------------------------

    rectangle(image_, bounding_rect, Scalar(0, 255, 0), 2);
    circle(image_, centor, 10, Scalar(255, 0, 0), -1);

    putText(image_, "Red_cross", Point(bounding_rect.x, bounding_rect.y - 10),
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);

    Point2d output;
    if (max_contours.empty()) {
        output.x = camera::NULL_ERROR;
        output.y = camera::NULL_ERROR;
    } else {
        output.x = static_cast<double>(bounding_rect.x + static_cast<double>(bounding_rect.width) / 2);
        output.y = static_cast<double>(bounding_rect.y + static_cast<double>(bounding_rect.height) / 2);
    }
    imshow("22222", mask);
    imshow("11111", image_);
    waitKey(30);
    return output;
}

#endif
#ifdef  USB_USE_YOLO
std::vector<DetectVector> usb_camera::USBFactor::Receive_Keypoint() {
    times++;
    vector<DetectorVector> output_points;
    if (times < 30) {
        return output_points;
    }

    image_ = usb_camera_->GetFrame();
    if (image_.empty()) {
        return output_points;
    }

    std::vector<Yolov8::Detection> detections = detector_.detect(image_);
    //std::vector<Yolov8::Detection> detections ;

    if (detections.empty()) {
        imshow("22222", image_);
        waitKey(30);
        return output_points;
    }

    for (auto &detection: detections) {
        DetectorVector detection_vector;
        Point point;
        point.x = detection.box.x + detection.box.width / 2;
        point.y = detection.box.y + detection.box.height / 2;

        circle(image_, point, 6, Scalar(255, 0, 0), -1);

        string label = format("%s: %.2f", ClassNames[detection.classId].c_str(), detection.confidence);
        int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        Rect labelRect = Rect(detection.box.x, detection.box.y - labelSize.height - baseLine,
                              labelSize.width, labelSize.height + baseLine);
        rectangle(image_, labelRect, Scalar(0, 255, 0), FILLED);
        putText(image_, label, Point(detection.box.x, detection.box.y - baseLine),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);

        // 所有类别都保留 (新模型无none类)
        //--------------------------------------------------------------------------------------------
        {
            detection_vector.point=point;
            detection_vector.ClassId = detection.classId;
            output_points.push_back(detection_vector);
        }
    }

    // for (auto &point: output_points) {
    //
    // }

    imshow("22222", image_);
    waitKey(30);

    return output_points;
}
#endif


Point2d usb_camera::USBFactor::Transform_Image_TO_Real(cv::Point2d &image_point,
                                                       geometry_msgs::msg::TransformStamped pose) {
    double height = pose.transform.translation.z + params_.height_offset; // 无人机高度 + 相机安装高度
    double theta_x, theta_z;
    double length;
    double cam_pitch;
    double real_x, real_y;

    /// --------------------------------转换在这里转的--------------------------------------------
    // double w = pose.transform.rotation.w;
    // double y = -pose.transform.rotation.x;
    // double x = pose.transform.rotation.y;
    // double z = pose.transform.rotation.z;

    double w = pose.transform.rotation.w;
    double x = pose.transform.rotation.x;
    double y = pose.transform.rotation.y;
    double z = pose.transform.rotation.z;

    std::cerr << "w=" << w << "  x=" << x << "  y=" << y << "  z=" << z << std::endl;

    std::cerr << "X=" << pose.transform.translation.x << "  Y=" << pose.transform.translation.y << "  Z=" << pose.
            transform.translation.z << "  height=" << height << "m" << std::endl;

    Eigen::Vector3d r_C; //相机系的向量
    Eigen::Quaterniond q(w, x, y, z); //构造的四元数
    Eigen::Matrix3d R_WD; //由四元数得到的旋转矩阵
    Eigen::Matrix3d R_DC; //相机对于无人机系的偏执
    Eigen::Matrix3d R_WC; //总的偏执矩阵
    Eigen::Vector3d r_W; //世界系中的相机向量
    Eigen::Vector3d target_W; //目标的世界向量

    cv::Point image_center(params_.cx(), params_.cy());
    theta_x = params_.horizon_x_rad() * (image_point.x - image_center.x) / image_center.x;
    theta_z = params_.horizon_z_rad() * (image_center.y - image_point.y) / image_center.y;

    //相机系方向向量
    r_C.x() = tan(theta_x);
    r_C.y() = tan(theta_z);
    r_C.z() = 1.0; //这里的 1.0 代表以1为单位高度
    r_C.normalize();

    //旋转矩阵（无人机系到世界系）
    R_WD = q.normalized().toRotationMatrix();

    //相机相对于无人机的固定旋转（从YAML读取）
    cam_pitch = params_.cam_pitch_rad();

    R_DC = Eigen::AngleAxisd(cam_pitch, Eigen::Vector3d::UnitX()) // roll
           * Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) // pitch
           * Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ()); // yaw

    //TFDebug tf2("camera","plane",R_DC);

    //相机系到世界系的变换
    R_WC = R_WD * R_DC;

#ifdef RVIZ_DEBUG
    tf_debug_ptr_->SetTransform(R_WC);
    tf_debug_ptr_->broadcast_transform();
#endif

    //方向向量转换到世界系
    r_W = R_WC * r_C;

    //计算目标在世界系中的位置
    double t = height / r_W.z();
    // if (t < 0) {
    //     return {camera::NULL_ERROR,camera::NULL_ERROR};
    // }
    target_W.x() = t * r_W.x();
    target_W.y() = t * r_W.y();
    target_W.z() = 0.0; //地面高度为0

    //转换到无人机系
    // Eigen::Vector3d drone_pos(pose.transform.translation.x,
    //                           pose.transform.translation.y,
    //                           pose.transform.translation.z);

    //Eigen::Vector3d target_D = R_WD.transpose() * (target_W - drone_pos);

    Eigen::Vector3d target_D = R_WD.transpose() * target_W;

    real_x = target_D.x();
    real_y = target_D.y();

    return {target_W.x(), target_W.y()};
}
