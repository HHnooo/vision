#ifndef KALMAN_KALMAN_HPP
#define KALMAN_KALMAN_HPP

#define QUEEN_LENGTH 20 //计算方差的窗口，窗口越大，滞后越大

#include <Eigen/Eigen>
#include <iostream>
#include <atomic>
#include <optional>
#include <cmath>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <rclcpp/rclcpp.hpp>

#include <Eigen/Dense>

namespace kalman {
    struct KalmanInput {
        double x; //图像像素x值
        double y; //图像像素y值
        double w; //框的x值
        double h; //框的y值
    };

    struct KalmanOutput {
        KalmanInput input;
        double v_x;
        double v_y;
        double v_w;
        double v_h;

        bool is_success;
        double sigma;
    };

    class TopicPublisher : public rclcpp::Node {
    public:
        TopicPublisher(const std::string &node_name,
                       const std::string &topic_name,
                       const rclcpp::QoS &qos_profile = rclcpp::QoS(10));

        ~TopicPublisher() override;

        void publish_message(const std::vector<double> &message);

    private:
        rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr publisher_;
    };

    class Kalman {
    private:
        uint8_t mode; //使用的卡尔曼模型，目前提供CA和CV模型
        uint16_t missing_cnt; //测量丢失计数
        uint16_t maxPredictCnt; //最大预测步长
        uint8_t noise_cnt; //连续噪声计数（用于区分噪声与突变）
        uint8_t stateSize; //状态量
        uint8_t measSize; //测量的状态量
        uint8_t uSize; //控制变量的维度
        double T; //卡尔曼周期
        uint8_t sigma_index; //不确定性计算的滑动窗口索引

        float numerator = (QUEEN_LENGTH - 1) * QUEEN_LENGTH * (QUEEN_LENGTH + 1) / 12.0f; //sigma计算常量
        float sigma_x[QUEEN_LENGTH] = {0};
        float sigma_y[QUEEN_LENGTH] = {0};


        float sum_x = 0;
        float sum_y = 0;

        float sum_x2 = 0;
        float sum_y2 = 0;


        float sum_ix = 0;
        float sum_iy = 0;

        Eigen::VectorXd x; //状态量
        Eigen::VectorXd z; //先验估计的观测量
        Eigen::VectorXd z_abnormal; //记录到的异常噪音
        Eigen::MatrixXd A; //状态转移矩阵
        Eigen::MatrixXd B; //控制矩阵
        Eigen::VectorXd u; //输入矩阵
        Eigen::MatrixXd P; //先验估计协方差
        Eigen::MatrixXd H; //观测矩阵
        Eigen::MatrixXd R; //测量噪声协方差
        Eigen::MatrixXd Q; //过程噪声协方差s
    public:
        explicit Kalman(uint8_t _mode, uint16_t _max_predict_cnt);
        //使用匀速运动还是加速运动
        enum KalmanType {
            CVMODE = 0,
            CAMODE = 1,
        };

        //初始化函数
        void init_x(Eigen::VectorXd &x_);

        //CV设置噪声协方差
        void Q_set_2d_CV(double q);

        //CA设置噪声协方差
        void Q_set_2d_CA(double qx);

        //测量协方差
        void R_set_2d(double rx);

        //初始化先验估计协方差
        void P_init(Eigen::MatrixXd &P_);

        //设置卡尔曼周期
        void T_set(double _T);

        //观测矩阵
        void H_set(Eigen::MatrixXd &H_);

        //设置状态转移矩阵
        void A_set(Eigen::MatrixXd &A_);

        //CV设置观测矩阵
        void H_set_2d_CV();

        //CV设置状态转移矩阵
        void A_set_2d_CV();

        //CA设置观测矩阵
        void H_set_2d_CA();

        //CA设置状态转移矩阵
        void A_set_2d_CA();

        float sigma_calculate(const Eigen::VectorXd &_wordCoord);

        float a_Var();

        Eigen::VectorXd predict(Eigen::MatrixXd &A_);

        Eigen::VectorXd predict();

        Eigen::VectorXd predict(Eigen::MatrixXd &A_, Eigen::MatrixXd &B_, Eigen::VectorXd &u_);

        Eigen::VectorXd update(const Eigen::VectorXd &z_meas);

        Eigen::VectorXd update(const Eigen::MatrixXd &H_, const Eigen::VectorXd &z_meas);

        Eigen::VectorXd update(const Eigen::VectorXd &z_meas, const Eigen::MatrixXd &_rota_matrix);

        KalmanOutput kalman_filter(std::atomic<bool> &is_unsuccessful, const Eigen::VectorXd &_wordCoord,
                                   const std::optional<Eigen::MatrixXd> &_rota_matrix = std::nullopt);

        bool is_large_noise(const Eigen::VectorXd &_current_wordCoord);
    };
}


#endif //KALMAN_KALMAN_HPP
