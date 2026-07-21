#ifndef CAMERA_DATA_HANDLE_PID_PREDICT_HPP
#define CAMERA_DATA_HANDLE_PID_PREDICT_HPP

#include <vector>
#include <chrono>
#include <tuple>
#include <opencv2/opencv.hpp>

#include <vector>
#include <chrono>
#include <tuple>
#include <Eigen/Dense>
#include <geometry_msgs/msg/detail/point_stamped__builder.hpp>
#include <geometry_msgs/msg/detail/transform_stamped__struct.hpp>
#include <geometry_msgs/msg/detail/twist__struct.hpp>

struct TargetPosition {
    double x; // X像素坐标
    double y; // Y像素坐标
    std::chrono::high_resolution_clock::time_point timestamp;

    TargetPosition(double x_, double y_)
        : x(x_), y(y_), timestamp(std::chrono::high_resolution_clock::now()) {
    }

    TargetPosition(double x_, double y_, std::chrono::high_resolution_clock::time_point ts)
        : x(x_), y(y_), timestamp(ts) {
    }
};

class TargetSpeedEstimator {
private:
    int img_width;
    int img_height;
    size_t max_history_size;
    double min_time_interval;
    double max_position_jump;
    double smoothing_factor;
    double current_vx;
    double current_vy;
    double max_integral;
    std::vector<TargetPosition> position_history;
    double kp;
    double ki;
    double kd;
    double integral_x;
    double integral_y;
    double prev_error_x;
    double prev_error_y;
    double history_weight_factor; // 0-1 值越大越相信当前速度

    //计算两个时间点之间的差值
    double calculate_time_diff(
        const std::chrono::high_resolution_clock::time_point &t1,
        const std::chrono::high_resolution_clock::time_point &t2);

    //判断新位置是否为异常值
    bool is_outlier(const TargetPosition &new_pos);

    // 平滑速度
    void smooth_speed(double &current, double new_value);

    // 计算历史速度的加权平均
    std::tuple<double, double> calculate_weighted_history_speed();

public:
    // 构造函数
    explicit TargetSpeedEstimator(int width = 640, int height = 480);

    // 设置历史记录最大数量
    void set_max_history_size(size_t size);

    // 设置最小时间间隔
    void set_min_time_interval(double interval);

    // 设置最大位置跳变阈值
    void set_max_position_jump(double jump);

    // 设置平滑因子
    void set_smoothing_factor(double factor);

    // 设置PID参数
    void set_pid_parameters(double p, double i, double d);

    // 设置历史权重因子
    void set_history_weight_factor(double factor);

    // 更新目标位置并计算速度
    void update_position(double x, double y);

    //使用PID控制获取平滑的目标速度
    std::tuple<double, double> get_smoothed_speed(double target_x, double target_y);

    [[nodiscard]] std::tuple<double, double> get_speed() const;

    [[nodiscard]] const std::vector<TargetPosition> &get_history() const;

    void set_kp(double p);

    void set_ki(double p);

    void set_kd(double p);

    void set_max_integral(double p);

    //重置所有状态
    void reset();
};


class V_Predict {
private:
    TargetSpeedEstimator estimator_;
    geometry_msgs::msg::Twist plane_velocity_; //无人机的速度向量
    geometry_msgs::msg::TransformStamped camera_pose_; //可以得到高度
    double rate_; // 速度转换的比率
    double dt_; //最后乘的系数

    std::tuple<double, double> Camera_Speed_To_Real(double height, double x, double y);

    std::tuple<double, double> Calculate_dv(double dx_1, double dy_1, double dx_2, double dy_2);

public:
    V_Predict();

    TargetSpeedEstimator get_estimator();

    cv::Point2f Predict(int x, int y,
                        geometry_msgs::msg::Twist twist,
                        geometry_msgs::msg::TransformStamped pose);

    cv::Point2f Predict(int x, int y);

    void set_rate(double rate) { rate_ = rate; }

    void set_dt(double dt) { dt_ = dt; }
};


#endif //CAMERA_DATA_HANDLE_PID_PREDICT_HPP
