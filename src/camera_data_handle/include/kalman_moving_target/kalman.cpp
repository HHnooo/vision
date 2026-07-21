#include "kalman.hpp"

/// sateSize状态量个数
/// uSize输入的维度

using namespace std;

//初始化kalman参数，第一个参数为使用的模型，第二个参数为最大预测步长，最大预测时间为：最大预测步长 * kalman运行周期
kalman::Kalman::Kalman(uint8_t _mode, uint16_t _max_predict_cnt)
    : mode(_mode),
      maxPredictCnt(_max_predict_cnt), T(0) {
    sigma_index = 0;
    noise_cnt = 0;
    missing_cnt = _max_predict_cnt;
    //missing_cnt = 0;

    switch (mode) {
        case CVMODE:
            stateSize = 8;
            measSize = 4;
            uSize = 0;
            break;
        case CAMODE:
            stateSize = 11;
            measSize = 4;
            uSize = 0;
            break;
        default:
            cerr << "Cannot Invalid Kalman mode" << endl;;
    }
    if (stateSize == 0 || measSize == 0) {
        std::cerr << "Error, State size and measurement size must bigger than 0\n";
    }
    x.resize(stateSize);
    x.setOnes();


    A.resize(stateSize, stateSize);
    A.setIdentity();

    u.resize(uSize);
    u.transpose();
    u.setZero();

    B.resize(stateSize, uSize);
    B.setZero();

    P.resize(stateSize, stateSize);
    P.setIdentity();

    H.resize(measSize, stateSize);
    H.setZero();

    z.resize(measSize);
    z.setZero();

    z_abnormal.resize(measSize);
    z_abnormal.setZero();

    Q.resize(stateSize, stateSize);
    Q.setZero();

    R.resize(measSize, measSize);
    R.setZero();

    T_set(0.001);

    switch (mode) {
        case KalmanType::CVMODE:
            A_set_2d_CV(); //设置CV模型的状态矩阵
            H_set_2d_CV(); //设置CV模型的测量矩阵
            Q_set_2d_CV(0.01); //设置CV模型的状态噪音
            R_set_2d(0.01); //设置CV模型的测量噪音
            break;
        case KalmanType::CAMODE:
            A_set_2d_CA(); //设置CV模型的状态矩阵
            H_set_2d_CA(); //设置CV模型的测量矩阵
            Q_set_2d_CA(0); //设置CV模型的状态噪音
            R_set_2d(0); //设置CV模型的测量噪音
            break;
    }
}

//初始化先验估计协方差
void kalman::Kalman::P_init(Eigen::MatrixXd &P_) {
    P = P_;
}


//初始化状态噪音
void kalman::Kalman::Q_set_2d_CV(double q) {
    Eigen::MatrixXd U;
    U.resize(measSize, measSize);
    U <<
            q, 0, 0, 0,
            0, q, 0, 0,
            0, 0, q * 0.5, 0,
            0, 0, 0, q * 0.5;
    Eigen::MatrixXd G;
    G.resize(stateSize, measSize);
    G <<
            0.5 * T * T, 0, 0, 0, //x受x噪声二次影响
            T, 0, 0, 0, //vx受x噪声一次影响
            0, 0.5 * T * T, 0, 0, //y受y噪声二次影响
            0, T, 0, 0, //vy受y噪声一次影响
            0, 0, 0.5 * T * T, 0, //w受w噪声二次影响
            0, 0, T, 0, //vw受w噪声一次影响
            0, 0, 0, 0.5 * T * T, //h受h噪声二次影响
            0, 0, 0, T; //vh受h噪声一次影响
    Q = G * U * G.transpose();
}

void kalman::Kalman::Q_set_2d_CA(double qx) {
    double Q_ax, Q_ay, Q_az;
    Q_ax = Q_ay = Q_az = qx;

    Eigen::MatrixXd U;
    U.resize(measSize, measSize);
    U << Q_ax, 0, 0,
            0, Q_ay, 0,
            0, 0, Q_az;
    Eigen::MatrixXd G;
    G.resize(stateSize, measSize);
    G <<
            pow(T, 3) / 6.0f, 0, 0,
            0.5f * pow(T, 2), 0, 0,
            T, 0, 0,
            0, pow(T, 3) / 6.0f, 0,
            0, 0.5f * pow(T, 2), 0,
            0, T, 0,
            0, 0, pow(T, 3) / 6.0f,
            0, 0, 0.5f * pow(T, 2),
            0, 0, T;
    Q = G * U * G.transpose();
}

//初始化测量噪音
void kalman::Kalman::R_set_2d(double r) {
    double R_x = r * r;
    double R_y = r * r;
    double R_w = (r * 0.6) * (r * 0.6);
    double R_h = (r * 0.6) * (r * 0.6);
    R <<
            R_x, 0, 0, 0,
            0, R_y, 0, 0,
            0, 0, R_w, 0,
            0, 0, 0, R_h;
}

//初始化，卡尔曼滤波初始化只需要初始化初始状态，初始协方差矩阵
void kalman::Kalman::init_x(Eigen::VectorXd &x_) {
    x = x_;
}

void kalman::Kalman::H_set(Eigen::MatrixXd &H_) {
    H = H_;
}

void kalman::Kalman::A_set(Eigen::MatrixXd &A_) {
    A = A_;
}

void kalman::Kalman::H_set_2d_CV() {
    H <<
            1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0; //设置测量矩阵
}

void kalman::Kalman::H_set_2d_CA() {
    H <<
            1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 0; //设置测量矩阵
}

void kalman::Kalman::A_set_2d_CV() {
    A <<
            1, T, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 1, T, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 1, T, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 1, T,
            0, 0, 0, 0, 0, 0, 0, 1; //设置状态矩阵
}

void kalman::Kalman::A_set_2d_CA() {
    A <<
            1, T, 0.5 * pow(T, 2), 0, 0, 0, 0, 0, 0,
            0, 1, T, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, T, 0.5 * pow(T, 2), 0, 0, 0,
            0, 0, 0, 0, 1, T, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, T, 0.5 * pow(T, 2),
            0, 0, 0, 0, 0, 0, 0, 1, T,
            0, 0, 0, 0, 0, 0, 0, 0, 1;
}

void kalman::Kalman::T_set(double _T) {
    T = _T;
}

float kalman::Kalman::sigma_calculate(const Eigen::VectorXd &_wordCoord) {
    static const int WINDOW_SIZE = 5; // 历史数据窗口大小
    static KalmanInput history[WINDOW_SIZE]; // 存储历史检测框
    static int window_idx = 0; // 窗口索引（循环覆盖）
    static bool is_window_full = false; // 窗口是否填满

    //更新历史缓存
    history[window_idx].x = _wordCoord[0];
    history[window_idx].y = _wordCoord[1];
    history[window_idx].w = _wordCoord[2];
    history[window_idx].h = _wordCoord[3];
    window_idx = (window_idx + 1) % WINDOW_SIZE;
    if (!is_window_full && window_idx == 0) {
        is_window_full = true; // 首次填满窗口
    }

    //若窗口未填满，返回默认值
    if (!is_window_full) {
        return 0.0f; //窗口未满时暂不计算
    }

    //对x、y、w、h四个维度分别进行线性拟合
    float t_sum = 0, t2_sum = 0; // 时间步总和、时间步平方和
    float x_sum = 0, y_sum = 0, w_sum = 0, h_sum = 0; // 检测值总和
    float tx_sum = 0, ty_sum = 0, tw_sum = 0, th_sum = 0; // t*检测值总和

    for (int t = 0; t < WINDOW_SIZE; t++) {
        //获取历史数据
        int idx = (window_idx + t) % WINDOW_SIZE;
        const KalmanInput &bbox = history[idx];

        t_sum += static_cast<float>(t);
        t2_sum += static_cast<float>(t * t);
        x_sum += static_cast<float>(bbox.x);
        y_sum += static_cast<float>(bbox.y);
        w_sum += static_cast<float>(bbox.w);
        h_sum += static_cast<float>(bbox.h);
        tx_sum += static_cast<float>(t * bbox.x);
        ty_sum += static_cast<float>(t * bbox.y);
        tw_sum += static_cast<float>(t * bbox.w);
        th_sum += static_cast<float>(t * bbox.h);
    }

    //计算线性拟合参数a（截距）和b（斜率）
    float n = WINDOW_SIZE;
    float denominator = n * t2_sum - t_sum * t_sum; //分母（避免除零）
    if (denominator < 1e-6) {
        return 0.0f;
    }

    //x方向拟合
    float b_x = (n * tx_sum - t_sum * x_sum) / denominator;
    float a_x = (x_sum - b_x * t_sum) / n;

    //y方向拟合
    float b_y = (n * ty_sum - t_sum * y_sum) / denominator;
    float a_y = (y_sum - b_y * t_sum) / n;

    //w方向拟合
    float b_w = (n * tw_sum - t_sum * w_sum) / denominator;
    float a_w = (w_sum - b_w * t_sum) / n;

    //h方向拟合
    float b_h = (n * th_sum - t_sum * h_sum) / denominator;
    float a_h = (h_sum - b_h * t_sum) / n;

    //计算每个维度的残差标准差
    float x_sigma = 0, y_sigma = 0, w_sigma = 0, h_sigma = 0;
    for (int t = 0; t < WINDOW_SIZE; t++) {
        int idx = (window_idx + t) % WINDOW_SIZE;
        const KalmanInput &bbox = history[idx];
        //计算当前值与拟合值的偏差平方
        x_sigma += static_cast<float>(pow(bbox.x - (a_x + b_x * (float) t), 2));
        y_sigma += static_cast<float>(pow(bbox.y - (a_y + b_y * (float) t), 2));
        w_sigma += static_cast<float>(pow(bbox.w - (a_w + b_w * (float) t), 2));
        h_sigma += static_cast<float>(pow(bbox.h - (a_h + b_h * (float) t), 2));
    }
    //标准差
    x_sigma = sqrt(x_sigma / n);
    y_sigma = sqrt(y_sigma / n);
    w_sigma = sqrt(w_sigma / n);
    h_sigma = sqrt(h_sigma / n);

    //综合四个维度的标准差，取最大值作为整体噪声水平
    return std::max({x_sigma, y_sigma, w_sigma, h_sigma});
}

float kalman::Kalman::a_Var() {
    //滑动窗口参数（可根据需求调整）
    static const int WINDOW_SIZE = 5; // 历史速度数据窗口大小
    //存储历史速度估计值 (vx, vy, vw, vh)
    static Eigen::Matrix4Xf speed_history = Eigen::Matrix4Xf::Zero(4, WINDOW_SIZE);
    static int window_idx = 0; // 窗口索引（循环覆盖）
    static bool is_window_full = false; // 窗口是否填满

    //从当前状态向量中提取速度估计值
    //状态向量定义: [x, vx, y, vy, w, vw, h, vh]
    auto vx = static_cast<float>(x(1)); // x方向速度
    auto vy = static_cast<float>(x(3)); // y方向速度
    auto vw = static_cast<float>(x(5)); // 宽度变化速度
    auto vh = static_cast<float>(x(7)); // 高度变化速度

    //更新历史速度缓存（循环覆盖最旧数据）
    speed_history.col(window_idx) << vx, vy, vw, vh;
    window_idx = (window_idx + 1) % WINDOW_SIZE;
    // if (!is_window_full && window_idx == 0) {
    //     is_window_full = true; // 首次填满窗口
    // }
    //
    // //若窗口未填满，返回默认值（避免初始阶段误差）
    // if (!is_window_full) {
    //     return 0.0f;
    // }

    //计算每个速度分量的均值
    Eigen::Vector4f speed_mean = speed_history.rowwise().mean();

    //计算每个速度分量的方差
    Eigen::Vector4f speed_var = Eigen::Vector4f::Zero();
    for (int i = 0; i < WINDOW_SIZE; ++i) {
        Eigen::Vector4f diff = speed_history.col(i) - speed_mean;
        speed_var += diff.array().square().matrix();
    }
    speed_var /= WINDOW_SIZE; //求平均方差

    //计算每个速度分量的标准差
    Eigen::Vector4f speed_std = speed_var.array().sqrt();

    return speed_std.maxCoeff();
}

//有输入矩阵和控制矩阵的卡尔曼滤波预测过程
Eigen::VectorXd kalman::Kalman::predict(Eigen::MatrixXd &A_, Eigen::MatrixXd &B_, Eigen::VectorXd &u_) {
    A = A_;
    B = B_;
    u = u_;
    x = A * x + B * u; //根据时刻t-1的状态由状态转换举着预测时刻t的先验估计值，
    P = A * P * A.transpose() + Q; //时刻t先验估计值的的协方差,Q为过程噪声的协方差
    return x;
}

//没有输入矩阵和控制矩阵的卡尔曼滤波预测过程
Eigen::VectorXd kalman::Kalman::predict(Eigen::MatrixXd &A_) {
    A = A_;
    x = A * x;
    P = A * P * A.transpose() + Q;
    //  cout << "P-=" << P<< endl;
    return x;
}

//预测模型不变
Eigen::VectorXd kalman::Kalman::predict() {
    x = A * x;
    P = A * P * A.transpose() + Q;
    //  cout << "P-=" << P<< endl;
    return x;
}

// H_   观测矩阵
//z_means 观测量，由系统真实观测输入
Eigen::VectorXd kalman::Kalman::update(const Eigen::MatrixXd &H_, const Eigen::VectorXd &z_meas) {
    H = H_;
    Eigen::MatrixXd Ht = H.transpose();
    Eigen::MatrixXd K = P * Ht * (H * P * Ht + R).inverse();
    z = H * x;
    x = x + K * (z_meas - z);
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(stateSize, stateSize); //Identity()单位阵
    P = (I - K * H) * P;
    //  cout << "P=" << P << endl;
    return x;
}

//测量模型不变
Eigen::VectorXd kalman::Kalman::update(const Eigen::VectorXd &z_meas) {
    Eigen::MatrixXd Ht = H.transpose();
    Eigen::MatrixXd K = P * Ht * (H * P * Ht + R).inverse();
    z = H * x;
    x = x + K * (z_meas - z);
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(stateSize, stateSize); //Identity()单位阵
    P = (I - K * H) * P;
    //  cout << "P=" << P << endl;
    return x;
}

//测量模型不变,噪音变化
Eigen::VectorXd kalman::Kalman::update(const Eigen::VectorXd &z_meas, const Eigen::MatrixXd &rota_matrix) {
    Eigen::MatrixXd R_rota = rota_matrix * R * rota_matrix.transpose(); //将相机噪音乘上旋转矩阵，获得整体噪音

    Eigen::MatrixXd Ht = H.transpose();
    Eigen::MatrixXd K = P * Ht * (H * P * Ht + R_rota).inverse();
    z = H * x;
    x = x + K * (z_meas - z);
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(stateSize, stateSize); //Identity()单位阵
    P = (I - K * H) * P;
    //  cout << "P=" << P << endl;
    return x;
}

kalman::KalmanOutput kalman::Kalman::kalman_filter(std::atomic<bool> &is_unsuccessful,
                                                   const Eigen::VectorXd &_wordCoord,
                                                   const std::optional<Eigen::MatrixXd> &_rota_matrix) {
    KalmanOutput output{};
    //判断四种情况
    if (missing_cnt >= maxPredictCnt && !is_unsuccessful) {
        Eigen::VectorXd x_init;
        x_init.resize(stateSize);
        switch (stateSize) {
            case 2:
                x_init << _wordCoord(0), 0;
                break;
            case 6:
                x_init << _wordCoord(0), 0, _wordCoord(1), 0, _wordCoord(2), 0;
                break;
            case 8:
                x_init << _wordCoord(0), 0, _wordCoord(1), 0, _wordCoord(2), 0, _wordCoord(3), 0;
                break;
            case 9:
                x_init << _wordCoord(0), 0, 0, _wordCoord(1), 0, 0, _wordCoord(2), 0, 0;
                break;
        }
        init_x(x_init);

        z_abnormal = _wordCoord; //初始化异常噪音
        missing_cnt = 0;
        output.is_success = false;
    } else if (!is_unsuccessful) {
        //识别成功
        predict();
        if (is_large_noise(_wordCoord)) //出现异常噪音
        {
            predict();
        } else {
            if (_rota_matrix.has_value()) {
                update(_wordCoord, _rota_matrix.value());
            } else {
                update(_wordCoord);
            }
            missing_cnt = 0;
            output.is_success = true;
        }
        //ROS_INFO("yuce,x:%f,vx:%f,y:%f,vy:%f,z:%f,vz:%f",kalman_x_vector(0),kalman_x_vector(1),kalman_x_vector(2),kalman_x_vector(3),kalman_x_vector(4),kalman_x_vector(5));
    } else if (missing_cnt < maxPredictCnt) {
        //不成功但是在KALMAN_PREDICT_MAX步内
        predict();
        missing_cnt++;
        output.is_success = true;
        //ROS_INFO("celiang,x:%f,vx:%f,y:%f,vy:%f,z:%f,vz:%f",kalman_x_vector(0),kalman_x_vector(1),kalman_x_vector(2),kalman_x_vector(3),kalman_x_vector(4),kalman_x_vector(5));
    } else {
        //丢失时间过长,判定丢失
        missing_cnt = maxPredictCnt;
        output.is_success = false;
    }

    switch (mode) {
        case KalmanType::CVMODE:
            output.input.x = x(0);
            output.input.y = x(2);
            output.input.w = x(4);
            output.input.h = x(6);
            output.v_x = x(1);
            output.v_y = x(3);
            output.v_w = x(5);
            output.v_h = x(7);

            output.sigma = sigma_calculate(_wordCoord);
            //std::cout << "kalman_CV ---------------------------------" << std::endl;
            break;
        case KalmanType::CAMODE:
            output.input.x = x(0);
            output.input.y = x(1);
            output.input.w = x(2);
            output.input.h = x(3);
            output.v_x = x(4);
            output.v_y = x(5);
            output.v_w = x(6);
            output.v_h = x(7);
            output.sigma = a_Var();
            //std::cout << "kalman_CA ---------------------------------" << std::endl;
            break;
        default:
            cerr << "-------------------------------------" << std::endl;
    }
    if (!is_unsuccessful) //用来清除前端的测量标志
    {
        is_unsuccessful = true;
    }

    return output;
}

bool kalman::Kalman::is_large_noise(const Eigen::VectorXd &_current_wordCoord) {
    switch (measSize) {
        case 4: {
            auto noise_x = static_cast<float>(abs(_current_wordCoord(0) - z_abnormal(0)));
            auto noise_y = static_cast<float>(abs(_current_wordCoord(1) - z_abnormal(1)));
            auto noise_w = static_cast<float>(abs(_current_wordCoord(2) - z_abnormal(2)));
            auto noise_h = static_cast<float>(abs(_current_wordCoord(3) - z_abnormal(3)));

            if (sqrt(noise_x * noise_x + noise_y * noise_y + noise_w * noise_w + noise_h * noise_h) > 16)
                noise_cnt++;
            else
                noise_cnt = 0;

            if (noise_cnt == 0) {
                z_abnormal = _current_wordCoord;
                return false;
            } else if (noise_cnt <= 5) //是噪音
                return true;
            else {
                //和上一次出现的位置偏差稳定，认为位置已经改变
                z_abnormal = _current_wordCoord;
                return false;
            }
        }
        default:
            return false;
    }
}

///-----------------------------------------------------------------------------------------------------------------------------------
/**
 *发布信息进行调试
*/

kalman::TopicPublisher::TopicPublisher(const std::string &node_name,
                                       const std::string &topic_name,
                                       const rclcpp::QoS &qos_profile)
    : Node(node_name) {
    // 创建发布者
    publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(topic_name, qos_profile);

    RCLCPP_INFO(this->get_logger(), "Topic publisher node '%s' initialized. Publishing to topic: '%s'",
                node_name.c_str(), topic_name.c_str());
}

kalman::TopicPublisher::~TopicPublisher() {
    RCLCPP_INFO(this->get_logger(), "Topic publisher node shutting down");
}

void kalman::TopicPublisher::publish_message(const std::vector<double> &message) {
    // 检查节点是否还在运行
    if (!rclcpp::ok()) {
        RCLCPP_WARN(this->get_logger(), "Node is not in a valid state, cannot publish message");
        return;
    }

    // 创建消息对象并填充数据
    auto msg = std_msgs::msg::Float64MultiArray();
    msg.data = message;

    // 发布消息
    publisher_->publish(msg);

    RCLCPP_DEBUG(this->get_logger(), "Published message");
}
