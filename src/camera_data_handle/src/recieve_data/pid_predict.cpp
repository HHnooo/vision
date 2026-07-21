#include "../../include/PID_predect/pid_predict.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>

#include <cmath>
#include <algorithm>
#include <stdexcept>

TargetSpeedEstimator::TargetSpeedEstimator(int width, int height)
    : img_width(width),
      img_height(height),
      max_history_size(10),
      min_time_interval(0.01), // 10毫秒
      max_position_jump(10.0), // 100像素
      smoothing_factor(0.1), // 平滑因子
      current_vx(0.0),
      current_vy(0.0),
      kp(0.5), // 默认PID参数
      ki(0.1),
      kd(0.2),
      integral_x(0.0),
      integral_y(0.0),
      prev_error_x(0.0),
      prev_error_y(0.0),
      history_weight_factor(0.7), // 默认更相信当前速度
      max_integral(100.0) {
    // 积分上限
}

// 历史记录大小设置接口
void TargetSpeedEstimator::set_max_history_size(size_t size) {
    if (size < 2) {
        throw std::invalid_argument("历史记录数量必须至少为2");
    }
    max_history_size = size;
    // 截断历史记录到新的最大尺寸
    while (position_history.size() > max_history_size) {
        position_history.erase(position_history.begin());
    }
}

// 最小时间间隔设置接口
void TargetSpeedEstimator::set_min_time_interval(double interval) {
    if (interval <= 0) {
        throw std::invalid_argument("时间间隔必须为正数");
    }
    min_time_interval = interval;
}

// 最大位置跳变阈值设置接口
void TargetSpeedEstimator::set_max_position_jump(double jump) {
    if (jump <= 0) {
        throw std::invalid_argument("位置跳变阈值必须为正数");
    }
    max_position_jump = jump;
}

// 平滑因子设置接口
void TargetSpeedEstimator::set_smoothing_factor(double factor) {
    if (factor < 0 || factor > 1) {
        throw std::invalid_argument("平滑因子必须在0到1之间");
    }
    smoothing_factor = factor;
}

// PID参数设置接口
void TargetSpeedEstimator::set_pid_parameters(double p, double i, double d) {
    if (p < 0 || i < 0 || d < 0) {
        throw std::invalid_argument("PID参数必须为非负数");
    }
    kp = p;
    ki = i;
    kd = d;
}

// 单独设置比例项参数
void TargetSpeedEstimator::set_kp(double p) {
    if (p < 0) {
        throw std::invalid_argument("比例参数必须为非负数");
    }
    kp = p;
}

// 单独设置积分项参数
void TargetSpeedEstimator::set_ki(double i) {
    if (i < 0) {
        throw std::invalid_argument("积分参数必须为非负数");
    }
    ki = i;
}

// 单独设置微分项参数
void TargetSpeedEstimator::set_kd(double d) {
    if (d < 0) {
        throw std::invalid_argument("微分参数必须为非负数");
    }
    kd = d;
}

// 历史权重因子设置接口
void TargetSpeedEstimator::set_history_weight_factor(double factor) {
    if (factor < 0 || factor > 1) {
        throw std::invalid_argument("历史权重因子必须在0到1之间");
    }
    history_weight_factor = factor;
}

// 积分上限设置接口
void TargetSpeedEstimator::set_max_integral(double max) {
    if (max <= 0) {
        throw std::invalid_argument("积分上限必须为正数");
    }
    max_integral = max;
}

// 计算时间差
double TargetSpeedEstimator::calculate_time_diff(
    const std::chrono::high_resolution_clock::time_point &t1,
    const std::chrono::high_resolution_clock::time_point &t2) {
    auto duration = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
    return duration.count();
}

// 判断是否为异常值
bool TargetSpeedEstimator::is_outlier(const TargetPosition &new_pos) {
    if (position_history.empty()) {
        return false; // 没有历史数据，无法判断
    }

    // 与最后一个位置比较
    const auto &last_pos = position_history.back();
    double dx = new_pos.x - last_pos.x;
    double dy = new_pos.y - last_pos.y;
    double distance = std::sqrt(dx * dx + dy * dy);

    // 计算时间差
    double dt = calculate_time_diff(last_pos.timestamp, new_pos.timestamp);

    // 如果时间差很小但位置变化很大，则认为是异常值
    if (dt < min_time_interval && distance > max_position_jump) {
        return true;
    }

    return false;
}

// 速度平滑处理
void TargetSpeedEstimator::smooth_speed(double &current, double new_value) {
    current = (1 - smoothing_factor) * current + smoothing_factor * new_value;
}

// 计算加权历史速度
std::tuple<double, double> TargetSpeedEstimator::calculate_weighted_history_speed() {
    if (position_history.size() < 2) {
        return std::make_tuple(0.0, 0.0);
    }

    // 计算历史速度的加权平均，最近的权重更高
    double sum_vx = 0.0, sum_vy = 0.0;
    double sum_weights = 0.0;
    int count = std::min(8, (int) position_history.size() - 1); // 最多使用8步历史

    // 从最近的开始计算，逐步向前取历史数据
    for (int i = 1; i <= count; ++i) {
        int idx = position_history.size() - 1 - i;
        if (idx < 0) break;

        double dt = calculate_time_diff(
            position_history[idx].timestamp,
            position_history[idx + 1].timestamp);

        if (dt < min_time_interval) continue;

        // 计算权重：越近的历史权重越高
        double weight = 1.0 / i; // 倒数权重，最近的权重最高
        sum_weights += weight;

        sum_vx += ((position_history[idx + 1].x - position_history[idx].x) / dt) * weight;
        sum_vy += ((position_history[idx + 1].y - position_history[idx].y) / dt) * weight;
    }

    if (sum_weights < 1e-6) {
        return std::make_tuple(0.0, 0.0);
    }

    return std::make_tuple(sum_vx / sum_weights, sum_vy / sum_weights);
}

// 更新位置信息
void TargetSpeedEstimator::update_position(double x, double y) {
    // 检查坐标是否在图像范围内
    if (x < 0 || x >= img_width || y < 0 || y >= img_height) {
        throw std::invalid_argument("目标位置超出图像范围");
    }

    // 创建新位置记录
    TargetPosition new_pos(x, y);

    // 检查是否为异常值
    if (is_outlier(new_pos)) {
        // 是异常值，不添加到历史记录，也不更新速度
        return;
    }

    // 添加到历史记录
    position_history.push_back(new_pos);

    // 保持历史记录大小
    while (position_history.size() > max_history_size) {
        position_history.erase(position_history.begin());
    }

    // 计算速度（需要至少两个位置）
    if (position_history.size() >= 2) {
        // 使用最近的两个位置计算原始速度
        const auto &prev_pos = position_history[position_history.size() - 2];
        double dt = calculate_time_diff(prev_pos.timestamp, new_pos.timestamp);

        // 确保时间差有效
        if (dt >= min_time_interval) {
            double raw_vx = (new_pos.x - prev_pos.x) / dt;
            double raw_vy = (new_pos.y - prev_pos.y) / dt;

            // 应用平滑
            smooth_speed(current_vx, raw_vx);
            smooth_speed(current_vy, raw_vy);
        }
    }
}

// 获取当前速度
std::tuple<double, double> TargetSpeedEstimator::get_speed() const {
    return std::make_tuple(current_vx, current_vy);
}

// 获取经过PID平滑的速度
std::tuple<double, double> TargetSpeedEstimator::get_smoothed_speed(double target_x, double target_y) {
    if (position_history.empty()) {
        return std::make_tuple(0.0, 0.0);
    }

    // 获取当前位置
    const auto &current_pos = position_history.back();

    // 计算误差
    double error_x = target_x - current_pos.x;
    double error_y = target_y - current_pos.y;

    // 计算时间差（与上一次计算的时间）
    static std::chrono::high_resolution_clock::time_point last_pid_time = current_pos.timestamp;
    double dt = calculate_time_diff(last_pid_time, std::chrono::high_resolution_clock::now());
    last_pid_time = std::chrono::high_resolution_clock::now();

    // 积分项
    integral_x += error_x * dt;
    integral_y += error_y * dt;
    integral_x = std::clamp(integral_x, -max_integral, max_integral);
    integral_y = std::clamp(integral_y, -max_integral, max_integral);

    // 微分项
    double derivative_x = dt > 1e-6 ? (error_x - prev_error_x) / dt : 0.0;
    double derivative_y = dt > 1e-6 ? (error_y - prev_error_y) / dt : 0.0;

    // 保存当前误差作为下一次的前项误差
    prev_error_x = error_x;
    prev_error_y = error_y;

    // 计算PID输出
    double pid_vx = kp * error_x + ki * integral_x + kd * derivative_x;
    double pid_vy = kp * error_y + ki * integral_y + kd * derivative_y;

    // 获取历史速度加权平均
    auto [history_vx, history_vy] = calculate_weighted_history_speed();

    // 结合当前PID速度和历史速度，使用权重因子
    double final_vx = history_weight_factor * pid_vx + (1 - history_weight_factor) * history_vx;
    double final_vy = history_weight_factor * pid_vy + (1 - history_weight_factor) * history_vy;

    return std::make_tuple(final_vx, final_vy);
}

// 获取历史记录
const std::vector<TargetPosition> &TargetSpeedEstimator::get_history() const {
    return position_history;
}

// 重置所有状态
void TargetSpeedEstimator::reset() {
    position_history.clear();
    current_vx = 0.0;
    current_vy = 0.0;
    integral_x = 0.0;
    integral_y = 0.0;
    prev_error_x = 0.0;
    prev_error_y = 0.0;
}

///----------------------------------------------------------------------------------------------------------------------------------------------------------

V_Predict::V_Predict() {
    estimator_.set_max_history_size(8);
    estimator_.set_smoothing_factor(0.1);
    estimator_.set_max_position_jump(30.0);
    rate_=1.0;
}

std::tuple<double, double> V_Predict::Calculate_dv(double dx_1, double dy_1, double dx_2, double dy_2) {
    return std::make_tuple(dx_1 - dx_2, dy_1 - dy_2);
}

std::tuple<double, double> V_Predict::Camera_Speed_To_Real(double height, double x, double y) {
    height += 0.15;
    double x1 = height * rate_ * x * 35 / (12 * 207);
    double y1 = height * rate_ * y * 35 / (12 * 207);
    std::cout << x1 << "    " << y1 << std::endl;
    return std::make_tuple(x1, y1); //h:105   x:36
}


cv::Point2f V_Predict::Predict(int x, int y, geometry_msgs::msg::Twist twist,
                               geometry_msgs::msg::TransformStamped pose) {
    estimator_.update_position(x, y);
    camera_pose_.transform = pose.transform;
    plane_velocity_ = twist;
    auto [camera_dx,camera_dy] = estimator_.get_speed();
    auto [plane_dx,plane_dy] = Camera_Speed_To_Real(camera_pose_.transform.translation.z, camera_dx, camera_dy);
    auto [dx,dy] = Calculate_dv(plane_dx, plane_dy, twist.linear.x, twist.linear.y);
    return {static_cast<float>(dx * dt_), static_cast<float>(dy * dt_)};
}

cv::Point2f V_Predict::Predict(int x, int y) {
    estimator_.update_position(x, y);
    auto [camera_dx,camera_dy] = estimator_.get_speed();
    auto [plane_dx,plane_dy] = Camera_Speed_To_Real(1, camera_dx, camera_dy);
    auto [dx,dy] = Calculate_dv(plane_dx, plane_dy, 0, 0);
    return {static_cast<float>(dx * dt_), static_cast<float>(dy * dt_)};
}

TargetSpeedEstimator V_Predict::get_estimator() {
    return estimator_;
}