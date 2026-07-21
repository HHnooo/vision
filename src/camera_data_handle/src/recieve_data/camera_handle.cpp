#include "../../include/camera_data_handle/camera_handle.h"

#include <geometry_msgs/msg/detail/pose_stamped__struct.hpp>

#include "../../include/camera_data_handle/target_predict.h"
#include "../../include/tf_publish/tf_publish.h"
#include "../../include/kalman_moving_target/kalman.hpp"

#include <ctime>
#include "rclcpp/rclcpp.hpp"
#include "yaml-cpp/yaml.h"
#include "sensor_msgs/msg/image.hpp"
#include "opencv4/opencv2/opencv.hpp"
#include "cv_bridge/cv_bridge.h"
#include <openvino/openvino.hpp>

#define STR(s) #s
#define MACRO_TO_STR(s) STR(s)

#ifdef KALMAN_OPEN
constexpr uint8_t KALMAN_MODEL = kalman::Kalman::CVMODE; // 选择模型：CVMODE(8维状态)/CAMODE(9维状态)
constexpr uint16_t MAX_PREDICT_STEP = 3; // 最大预测步数（无观测时最多预测5次）
constexpr double KALMAN_PERIOD = 0.05; // 卡尔曼运行周期（50ms，即20Hz，匹配传感器频率）
constexpr double PROCESS_NOISE = 0.0001; // 过程噪声（模型不确定性，值越小越信任模型）
constexpr double MEAS_NOISE = 0.001; // 测量噪声（观测不确定性，值越小越信任传感器）
#endif

kalman::KalmanInput camera::transform_to_karman_input(const Yolov8::Detection &detection) {
    kalman::KalmanInput kalman{};
    kalman.x = detection.box.x;
    kalman.y = detection.box.y;
    kalman.w = detection.box.width;
    kalman.h = detection.box.height;
    return kalman;
}

Eigen::VectorXd camera::transform_to_eigen_vector(const kalman::KalmanInput &input) {
    Eigen::VectorXd eigen_vector;
    eigen_vector = Eigen::VectorXd::Zero(4);
    eigen_vector << input.x, input.y, input.w, input.h;
    return eigen_vector;
}

camera::ReceiveData::ReceiveData() : Node("receive_data"),
                                     detector_(
                                         MACRO_TO_STR(PROJECT_PATH)"/model/best.xml",
                                         Yolov8::defaultClassNames,
                                         640,
                                         0.7,
                                         0.4,
                                         640,
                                         480) {
    RCLCPP_INFO(this->get_logger(), "Receive Data");
    RCLCPP_INFO(this->get_logger(), MACRO_TO_STR(PROJECT_PATH)"/model/best.xml");

    auto t = time(nullptr);
    char ts[64];
    strftime(ts, sizeof(ts), "detect_%Y%m%d_%H%M%S.avi", localtime(&t));
    video_name_ = string(MACRO_TO_STR(PROJECT_PATH)) + "/../../vidios/" + ts;

#ifdef VIDEO_WRITE
    writer_.open(video_name_,
                 cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                 15.0,
                 Size(640, 480),
                 true
    );
#endif

#ifdef KALMAN_OPEN
    kf_ = std::make_shared<kalman::Kalman>(KALMAN_MODEL, MAX_PREDICT_STEP); //创建对象
    kf_->T_set(KALMAN_PERIOD); //设置运行周期
    kf_->Q_set_2d_CV(PROCESS_NOISE); //设置过程噪声
    kf_->R_set_2d(MEAS_NOISE); //设置观测噪声
    Eigen::MatrixXd init_P(8, 8);
    init_P.setIdentity();
    init_P *= 8.0;
    kf_->P_init(init_P);
#ifdef KALMAN_OPEN_DEBUG
    measure_pub_ = this->create_publisher<Measure>("measure", 10);
    kalman_pub_ = this->create_publisher<KalmanOutput>("kalman", 10);
#endif
#endif
#ifdef PID_PREDICT_OPEN
    double max_history_size = 5.0;
    double smoothing_factor = 0.1;
    double max_position_jump = 30.0;
    double min_time_interval = 0.03;
    double pid_parameters_p = 0.5;
    double pid_parameters_i = 0.1;
    double pid_parameters_d = 0.2;
    double history_weight_factor = 0.7;
    double max_integral = 40.0;
    double estimator_rate_ = 1.0;
    double estimator_dt_ = 10.0;

    this->declare_parameter("max_history_size", 5.0);
    this->declare_parameter("smoothing_factor", 0.8);
    this->declare_parameter("max_position_jump", 30.0);
    this->declare_parameter("min_time_interval", 0.03);
    this->declare_parameter("pid_parameters_p", 0.5);
    this->declare_parameter("pid_parameters_i", 0.1);
    this->declare_parameter("pid_parameters_d", 0.2);
    this->declare_parameter("history_weight_factor", 0.7);
    this->declare_parameter("max_integral", 40.0);
    this->declare_parameter("estimator_rate_", 1.0);
    this->declare_parameter("estimator_dt_", 10.0);

    this->get_parameter("max_history_size", max_history_size);
    this->get_parameter("smoothing_factor", smoothing_factor);
    this->get_parameter("max_position_jump", max_position_jump);
    this->get_parameter("min_time_interval", min_time_interval);
    this->get_parameter("pid_parameters_p", pid_parameters_p);
    this->get_parameter("pid_parameters_i", pid_parameters_i);
    this->get_parameter("pid_parameters_d", pid_parameters_d);
    this->get_parameter("history_weight_factor", history_weight_factor);
    this->get_parameter("max_integral", max_integral);
    this->get_parameter("estimator_rate_", estimator_rate_);
    this->get_parameter("estimator_dt_", estimator_dt_);

    v_predict_.get_estimator().set_max_history_size(static_cast<size_t>(max_history_size));
    v_predict_.get_estimator().set_smoothing_factor(smoothing_factor);
    v_predict_.get_estimator().set_max_position_jump(max_position_jump);
    v_predict_.get_estimator().set_min_time_interval(min_time_interval);
    v_predict_.get_estimator().set_kp(pid_parameters_p);
    v_predict_.get_estimator().set_ki(pid_parameters_i);
    v_predict_.get_estimator().set_kd(pid_parameters_d);
    v_predict_.get_estimator().set_history_weight_factor(history_weight_factor);
    v_predict_.get_estimator().set_max_integral(max_integral);
    v_predict_.set_rate(estimator_rate_);
    v_predict_.set_dt(estimator_dt_);

    measure_pub_ = this->create_publisher<Measure>("measure", 10);
#endif
    // this->declare_parameter("nms_threshold_", 0.4);
    // this->declare_parameter("confidence_threshold_", 0.5);
    //
    // this->get_parameter("confidence_threshold_", confidence_threshold_);
    // this->get_parameter("nms_threshold_", nms_threshold_);
    position_pub_ = this->create_publisher<robot_interfaces::msg::ImageLocation>("/camera/target/position", 10);

    camera_switch_sub_ = this->create_subscription<std_msgs::msg::Bool>(
        "/camera/choose",
        10,
        [this](const std_msgs::msg::Bool::ConstSharedPtr &msg) {
            if_camera_enabled_ = msg->data;
        }
    );
    pose_sub_ = this->create_subscription<geometry_msgs::msg::TransformStamped>(
        "/robot/current_pose",
        10,
        [this](const geometry_msgs::msg::TransformStamped::ConstSharedPtr msg) {
            pose_.transform = msg->transform;
        }
    );
    image_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera/camera/color/image_raw",
        10,
        std::bind(&ReceiveData::imageCallback, this, std::placeholders::_1)
    );
    timer_ = this->create_wall_timer(
        std::chrono::seconds(1), [this]() {
            if (!has_received_) {
                RCLCPP_WARN(this->get_logger(), "No Image Receive");
            } else {
                RCLCPP_INFO(this->get_logger(), "Has been receive!");
                timer_->cancel();
            }
        }
    );
}

camera::ReceiveData::~ReceiveData() {
#ifdef VIDEO_WRITE
    if (writer_.isOpened()) {
        writer_.release();
        RCLCPP_INFO(this->get_logger(), "检测视频已保存到: %s", video_name_.c_str());
    }
#endif
}

void camera::ReceiveData::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
    has_received_ = true;
    try {
        fps_sum_++;
        if (msg->width <= 0 || msg->height <= 0) {
            RCLCPP_WARN(this->get_logger(), "Invalid image dimensions: %dx%d", msg->width, msg->height);
            return;
        }

#ifdef FPS_VISABLE_OPEN
        fps_timer_.start(); //计时以计算帧率

#endif

        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
        cv::Mat temp_image = cv_ptr->image;
        cv::Mat image;
        cv::cvtColor(temp_image, image, cv::COLOR_BGR2RGB);

#ifdef VIDEO_WRITE
// 移到 drawDetections 之后，确保写入的是带检测框的图像
#endif

        // 相机未使能时跳过目标检测
        if (!if_camera_enabled_) {
            RCLCPP_INFO(this->get_logger(), "No camera in image");
            return;
        }

#ifndef YOLOV8_DETECTOR_OFF
        //目标检测接口
        std::vector<Yolov8::Detection> detections = detector_.detect(image);
#ifdef KEY_POINT_TRACKING
        if (!detections.empty()) if_do_tracking_ = true;

        if (if_do_tracking_ && detections.empty()) {
            detections = detector_.track(image);
            if (detections.empty()) if_do_tracking_ = false;
        }
#endif
#ifdef PID_PREDICT_OPEN
        //Measure msg_;
        if (!detections.empty()) {
            estimator.update_position(detections.front().box.x, detections.front().box.y);
            //auto [dx,dy] = pid_tracker_.pid_control(detections.front().box.x, detections.front().box.y);
            auto [dx,dy] = estimator.get_speed();
            predict_result_ = v_predict_.Predict(detections.front().box.x, detections.front().box.y);
            // msg_.x = dx;
            // msg_.y = dy;
            // msg_.h = dx1;
            // msg_.w = dy1;
            circle(image,
                   Point2f(predict_result_.x + detections.front().box.x + detections.front().box.width / 2
                           , predict_result_.y + detections.front().box.y + detections.front().box.height / 2),
                   3, Scalar(255, 0, 0), 5);
        } else {
            max_running++;
        }
        if (max_running == 10) {
            max_running = 0;
            estimator.reset();
        }
        //measure_pub_->publish(msg_);
#endif
#ifdef KALMAN_OPEN
        if (!detections.empty() || kalman_step_ != 0) {
            kalman::KalmanInput kalman = transform_to_karman_input(detections.at(0));
            current_meas_ = transform_to_eigen_vector(kalman);
            if (!detections.empty()) is_meas_unsuccessful_ = false; //观测成功
            else {
                is_meas_unsuccessful_ = true; //观测失败
                kalman_step_++;
                if (kalman_step_ == 5) kalman_step_ = 0;
            }

            Measure measure;
            measure.x = current_meas_[0] + current_meas_[2] / 2.0;
            measure.y = current_meas_[1] + current_meas_[3] / 2.0;
            measure.w = current_meas_[2];
            measure.h = current_meas_[3];
            measure.id = detections.empty() ? UNKNOW : static_cast<uint8_t>(detections.at(0).classId);
#ifdef KALMAN_OPEN_DEBUG
        measure_pub_->publish(measure);
#endif
        kf_result_ = kf_->kalman_filter(
            is_meas_unsuccessful_,
            current_meas_,
            std::nullopt
        );

        KalmanOutput output;
        output.x = kf_result_.input.x;
        output.y = kf_result_.input.y;
        output.w = kf_result_.input.w;
        output.h = kf_result_.input.h;
        output.vx = kf_result_.v_x;
        output.vy = kf_result_.v_y;
        output.vw = kf_result_.v_w;
        output.vh = kf_result_.v_h;
        output.id = detections.empty() ? UNKNOW : static_cast<uint8_t>(detections.at(0).classId);
#ifdef KALMAN_OPEN_DEBUG
        kalman_pub_->publish(output);
#endif

        Rect kalman_box;
        kalman_box.x = static_cast<int>(kf_result_.input.x);
        kalman_box.y = static_cast<int>(kf_result_.input.y);
        kalman_box.width = static_cast<int>(kf_result_.input.w);
        kalman_box.height = static_cast<int>(kf_result_.input.h);
        cv::rectangle(image, kalman_box, Scalar(255, 0, 0), 2);
        std::cout << "  状态（x,y,w,h）：" << kf_result_.input.x << ", "
                << kf_result_.input.y << ", " << kf_result_.input.w << ", " << kf_result_.input.h << std::endl;
        std::cout << "  速度（vx,vy,vw,vh）：" << kf_result_.v_x << ", "
                << kf_result_.v_y << ", " << kf_result_.v_w << ", " << kf_result_.v_h << std::endl;
        std::cout << "  噪声指标（sigma）：" << kf_result_.sigma << " | 滤波有效？"
                << (kf_result_.is_success ? "是" : "否") << std::endl;
        }

#endif
        std::vector<std::vector<double> > positions = detector_.drawDetections(image, detections);

#ifdef VIDEO_WRITE
        writer_.write(image);
#endif
#endif


#ifdef YOLOV8_DETECTOR_OFF

        std::vector<std::vector<double> > positions{};
#endif

#ifdef PREDICT_OPEN
        //卡尔曼滤波器接口
        if (!detections.empty()) {
            tracking_num_ = 1;
        }
        try {
            if (tracker_ == nullptr && tracking_num_) {
                tracker_ = new KalmanBoxTracker(detections.front().box);
            } else if (tracking_num_) {
                if (!detections.empty()) {
                    //RCLCPP_ERROR(this->get_logger(), "-----------------------------!!!!!");
                    Rect input_rect = detections.front().box;
                    tracker_->update(input_rect);
                }
                Point2f predictCenter = tracker_->predict();
                Rect estimatedBox = tracker_->get_state();
        //RCLCPP_ERROR(this->get_logger(), "-----------------------------");

#ifndef NO_IMAGE
        circle(image, predictCenter, 5, Scalar(200, 0, 120), 2);
        cv::rectangle(image, estimatedBox, Scalar(255, 0, 0), 2);
#endif

        // cv::Point2d new_center = target_predict_factory_.StartPredict({predictCenter.x, predictCenter.y},
        //                                                               std::chrono::system_clock::now().
        //                                                               time_since_epoch().count());
        // circle(image, new_center, 5, Scalar(200, 20, 120), 2);
            }
        } catch (cv::Exception e) {
            RCLCPP_WARN(this->get_logger(), "%s", e.what());
        }

        if (detections.empty() && tracking_num_) {
            tracking_num_++;
            if (tracking_num_ == 10) {
                tracking_num_ = 0;
            }
        }

#endif

        //信息发送变量
        robot_interfaces::msg::ImageLocation position_msg;

        if (positions.empty()) {
            RCLCPP_WARN(this->get_logger(), "No detections found");
            position_msg.image_x = 0;
            position_msg.image_y = 0;
            position_msg.id = UNKNOW;
            position_pub_->publish(position_msg);
        } else {
            // 遍历所有检测结果，每个目标发布一条消息，带上各自的类别ID
            for (size_t i = 0; i < detections.size() && i < positions.size(); i++) {
                RCLCPP_INFO(this->get_logger(), "Detected [%s] classId=%d conf=%.2f at (%.1f, %.1f)",
                            Yolov8::defaultClassNames[detections[i].classId].c_str(),
                            detections[i].classId,
                            detections[i].confidence,
                            positions[i].at(0), positions[i].at(1));
#ifndef PUB_PRE_NOT_REAL
                position_msg.image_x = static_cast<float>(positions[i].at(0));
                position_msg.image_y = static_cast<float>(positions[i].at(1));
#endif
#ifdef PUB_PRE_NOT_REAL
                position_msg.image_x = static_cast<float>(predict_result_.x);
                position_msg.image_y = static_cast<float>(predict_result_.y);
#endif
                position_msg.id = detections[i].classId;
                position_pub_->publish(position_msg);
            }
        }
#ifndef NO_IMAGE
        // cv::circle(image, cv::Point(320, 240), 207, cv::Scalar(0, 255, 0), 1);
        // cv::line(image, cv::Point(0, 240), cv::Point(640, 240), cv::Scalar(0, 255, 0), 1);
        // cv::line(image, cv::Point(320, 0), cv::Point(320, 480), cv::Scalar(0, 255, 0), 1);
        cv::circle(image, cv::Point(320, 240), 207, cv::Scalar(0, 255, 0), 1);
        cv::line(image, cv::Point(0, 240), cv::Point(640, 240), cv::Scalar(0, 255, 0), 1);
        cv::line(image, cv::Point(320, 0), cv::Point(320, 480), cv::Scalar(0, 255, 0), 1);
#endif

#ifdef FPS_VISABLE_OPEN
        //计算fps的
        {
            fps_timer_.stop();
            double currentTime = fps_timer_.getTimeSec();
            fps_time_sum_ += currentTime;
            double fps = fps_sum_ / fps_time_sum_;
            std::stringstream ss;
            ss << "FPS: " << std::fixed << std::setprecision(2) << fps;
            cv::putText(image, ss.str(), cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        }
#endif

#ifndef NO_IMAGE
        namedWindow("image", cv::WINDOW_NORMAL);
        cv::resizeWindow("image", 1000, 750);
#endif

#ifdef FPS_VISABLE_OPEN
        fps_timer_.reset();

        if (fps_sum_ == 20) {
            fps_sum_ = 0;
            fps_time_sum_ = 0;
        }
#endif

#ifndef NO_IMAGE
        cv::imshow("image", image);
        cv::waitKey(1);
#endif
    } catch (cv_bridge::Exception &e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    } catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in image callback: %s", e.what());
    } catch (...) {
        RCLCPP_ERROR(this->get_logger(), "Unknown exception in image callback");
    }
}