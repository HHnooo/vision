#include "kalman.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <random>
#include <thread>

const uint8_t KALMAN_MODEL = kalman::Kalman::CVMODE;
const uint16_t MAX_PREDICT_STEP = 5;
const double KALMAN_PERIOD = 0.05; //卡尔曼运行周期（50ms，即20Hz，匹配传感器频率）
const double PROCESS_NOISE = 0.02; //过程噪声（模型不确定性，值越小越信任模型）
const double MEAS_NOISE = 0.8; //测量噪声（观测不确定性，值越小越信任传感器）
const int SIM_TOTAL_STEP = 60; //总模拟步数（60步×50ms=3秒，覆盖完整场景）

/**
 * 模拟目标运动的观测数据（x/y匀速移动，w/h缓慢变化，叠加小噪声）
 * @param step 当前时间步
 * @return 4维观测向量 [x(中心x), y(中心y), w(宽度), h(高度)]
 */
Eigen::VectorXd generate_sim_measurement(int step) {
    Eigen::VectorXd meas(4); // 观测向量固定4维（x,y,w,h）

    // 1. 基础运动规律（模拟目标匀速移动+尺寸稳定）
    double base_x = 5.0 + 0.6 * step; // x方向：初始5.0，每步移动0.6单位
    double base_y = 3.0 + 0.4 * step; // y方向：初始3.0，每步移动0.4单位
    double base_w = 10.0 + 0.02 * step; // 宽度：初始10.0，每步缓慢增大0.02
    double base_h = 15.0 + 0.03 * step; // 高度：初始15.0，每步缓慢增大0.03

    // 2. 叠加随机噪声（模拟传感器误差，符合正态分布）
    static std::default_random_engine gen;
    static std::normal_distribution<double> noise(0.0, 0.2); // 噪声均值0，标准差0.2

    meas << base_x + noise(gen), // x+噪声
            base_y + noise(gen), // y+噪声
            base_w + noise(gen), // w+噪声
            base_h + noise(gen); // h+噪声

    return meas;
}

// -------------------------- 辅助函数：模拟观测丢失场景 --------------------------
/**
 * 模拟观测丢失（每20步丢失3步，模拟传感器临时故障或目标遮挡）
 * @param step 当前时间步
 * @return true=观测丢失，false=观测正常
 */
bool simulate_meas_missing(int step) {
    // 第15-17步、35-37步、55-57步模拟丢失（覆盖不同阶段）
    return (step % 20 >= 15 && step % 20 <= 17);
}

// -------------------------- 主函数：卡尔曼滤波调用流程 --------------------------
int main() {
    // 1. 初始化卡尔曼滤波器
    kalman::Kalman kf(KALMAN_MODEL, MAX_PREDICT_STEP); // 模型类型+最大预测步数

    // 2. 配置卡尔曼核心参数（周期、噪声、初始协方差）
    // 2.1 设置运行周期（关键：状态矩阵A的计算依赖周期T）
    kf.T_set(KALMAN_PERIOD);

    // 2.2 设置过程噪声Q（根据模型选择对应函数）
    kf.Q_set_2d_CV(PROCESS_NOISE);

    // 2.3 设置测量噪声R（观测噪声，x/y权重高于w/h）
    kf.R_set_2d(MEAS_NOISE);

    // 2.4 设置初始协方差矩阵P（初始状态的不确定性，对角线值越大越不确定）
    Eigen::MatrixXd init_P(8, 8);
    init_P.setIdentity(); // 单位矩阵基础上放大不确定性
    init_P *= 8.0; // 初始不确定性：所有状态的协方差为8
    kf.P_init(init_P);

    // 3. 滤波流程变量初始化
    std::atomic<bool> is_meas_unsuccessful(true); // 观测是否失败（原子变量：线程安全）
    Eigen::VectorXd current_meas(4); // 当前观测值（x,y,w,h）
    kalman::KalmanOutput kf_result; // 滤波输出结果（状态+速度+噪声指标）

    // 4. 循环执行卡尔曼滤波（预测+更新）
    std::cout << "===================== 卡尔曼滤波开始（模型："
            << (KALMAN_MODEL == 0 ? "CV恒速" : "CA恒加速") << "）=====================" << std::endl;

    for (int step = 0; step < SIM_TOTAL_STEP; step++) {
        // 4.1 生成当前步观测数据（或标记丢失）
        bool is_missing = simulate_meas_missing(step);
        if (!is_missing) {
            current_meas = generate_sim_measurement(step); // 生成有效观测
            is_meas_unsuccessful = false; // 标记：观测成功
        } else {
            is_meas_unsuccessful = true; // 标记：观测失败（仅预测）
        }

        // 4.2 核心调用：执行卡尔曼滤波（无旋转矩阵，传std::nullopt）
        kf_result = kf.kalman_filter(
            is_meas_unsuccessful,
            current_meas,
            std::nullopt // 若需处理相机旋转，此处传入Eigen::MatrixXd类型旋转矩阵
        );

        // 4.3 输出当前步结果（清晰展示观测、滤波状态、速度、噪声）
        std::cout << "\n【第" << step + 1 << "步】" << std::endl;
        // 观测数据
        if (!is_missing) {
            std::cout << "观测数据：x=" << std::fixed << std::setprecision(2) << current_meas(0)
                    << ", y=" << current_meas(1) << ", w=" << current_meas(2) << ", h=" << current_meas(3) << std::endl;
        } else {
            std::cout << "观测数据：【丢失】（仅执行预测）" << std::endl;
        }
        // 滤波结果
        std::cout << "滤波结果：" << std::endl;
        std::cout << "  状态（x,y,w,h）：" << kf_result.input.x << ", "
                << kf_result.input.y << ", " << kf_result.input.w << ", " << kf_result.input.h << std::endl;
        std::cout << "  速度（vx,vy,vw,vh）：" << kf_result.v_x << ", "
                << kf_result.v_y << ", " << kf_result.v_w << ", " << kf_result.v_h << std::endl;
        std::cout << "  噪声指标（sigma）：" << kf_result.sigma << " | 滤波有效？"
                << (kf_result.is_success ? "是" : "否") << std::endl;

        // 4.4 模拟真实时间间隔（按卡尔曼周期休眠，匹配实际运行节奏）
        std::this_thread::sleep_for(std::chrono::duration<double>(KALMAN_PERIOD));
    }

    std::cout << "\n===================== 卡尔曼滤波结束 ======================" << std::endl;
    return 0;
}
