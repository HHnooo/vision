#include <utility>

#include "../../include/receive_openmv_data/receive_openmv_data.h"
#include "tf2/convert.hpp"
#include "tf2_eigen/tf2_eigen.hpp"

ReceiveOpenMVData::ReceiveOpenMVData(const std::string port,
                                     const speed_t baudRate) : port_(port), baudRate_(baudRate) {
}

std::string ReceiveOpenMVData::Receive_Openmv_Data() {
    int fd = 0;
    try {
        fd = openSerialPort(port_, baudRate_);
    } catch (const std::exception &e) {
        std::cerr << "A 发生错误: " << e.what() << std::endl;
    }

    int num = 0;

    try {
        while (num <= 10) {
            std::string line = readLine(fd);

            uint8_t byte_data;
            ssize_t n = read(fd, &byte_data, 1);

            // if (n > 0) {
            //     std::cout << "收到字节: 0x" << std::hex << static_cast<int>(byte_data) << std::dec << std::endl;
            //
            //     std::string bit_string;
            //     for (int i = 7; i >= 0; --i) {
            //         bit_string += ((byte_data >> i) & 1) ? '1' : '0';
            //     }
            //     std::cout << "按位解码: " << bit_string << std::endl;
            //     std::cout << "---------------------------" << std::endl;
            // }

            if (!line.empty()) {
                std::cout << "收到信息: " << line << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                return line;
            }
            num++;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    } catch (const std::exception &e) {
        std::cerr << "B 发生错误: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "程序已停止" << std::endl;
    }

    close(fd);
    std::cout << "串口已关闭" << std::endl;

    return "0";
}

int ReceiveOpenMVData::openSerialPort(const std::string &port, speed_t baudRate) {
    struct termios tty;
    int fd = -1;
    int retryCount = 0;
    constexpr int maxRetries = 10;

    while (retryCount < maxRetries) {
        fd = open(port.c_str(), O_RDWR | O_NOCTTY);
        if (fd == -1) {
            std::cerr << "无法打开串口: " << port << "，错误: " << strerror(errno) << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            retryCount++;
            continue;
        }

        if (!isatty(fd)) {
            std::cerr << port << " 不是一个终端设备" << std::endl;
            close(fd);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            retryCount++;
            continue;
        }

        if (tcgetattr(fd, &tty) == -1) {
            std::cerr << "无法获取串口属性，错误: " << strerror(errno) << std::endl;
            close(fd);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            retryCount++;
            continue;
        }

        break;
    }

    if (fd == -1 || retryCount >= maxRetries) {
        std::cerr << "达到最大重试次数，无法打开串口" << std::endl;
        return -1;
    }

    memset(&tty, 0, sizeof(tty));

    if (cfsetospeed(&tty, baudRate) == -1 || cfsetispeed(&tty, baudRate) == -1) {
        std::cerr << "无法设置波特率，错误: " << strerror(errno) << std::endl;
        close(fd);
        return -1;
    }

    tty.c_cflag &= ~PARENB; // 禁用奇偶校验
    tty.c_cflag &= ~CSTOPB; // 1停止位
    tty.c_cflag &= ~CSIZE; // 清除数据位设置
    tty.c_cflag |= CS8; // 8位数据位

    // 禁用硬件流控制
    tty.c_cflag &= ~CRTSCTS;

    // 启用接收器，设置本地模式
    tty.c_cflag |= (CLOCAL | CREAD);

    // 禁用软件流控制
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);

    // 禁用输入处理（不转换回车换行等）
    tty.c_iflag &= ~(ICRNL | INLCR | IGNCR);

    // 设置原始输入模式
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);

    // 设置原始输出模式
    tty.c_oflag &= ~OPOST;
    tty.c_oflag &= ~(ONLCR | OCRNL);

    // 调整超时设置 - 等待至少1个字符，超时1秒
    tty.c_cc[VTIME] = 10; // 1秒超时 (10 * 0.1秒)
    tty.c_cc[VMIN] = 1; // 至少读取1个字符

    // 清空输入输出缓冲区
    tcflush(fd, TCIFLUSH);

    // 应用配置
    if (tcsetattr(fd, TCSANOW, &tty) == -1) {
        std::cerr << "无法设置串口属性，错误: " << strerror(errno) << std::endl;
        close(fd);
        return -1;
    }

    return fd;
}

std::string ReceiveOpenMVData::readLine(int fd) {
    if (fd < 0) {
        return "";
    }

    std::string line;
    char c;
    ssize_t n;

    while (true) {
        n = read(fd, &c, 1);

        if (n == -1) {
            // 处理读取错误
            if (errno != EAGAIN && errno != EWOULDBLOCK) {
                std::cerr << "读取串口错误: " << strerror(errno) << std::endl;
            }
            break;
        } else if (n == 0) {
            // 没有读取到数据，超时
            break;
        } else {
            // 处理换行符和回车符
            if (c == '\n' || c == '\r') {
                if (!line.empty()) {
                    // 避免空行
                    break;
                }
                // 如果是空行则继续读取
                continue;
            }
            line += c;
        }
    }

    return line;
}

///CalculateTarget类----------------------------------------------------------------------------------------------------
#ifdef RVIZ_DEBUG
CalculateTarget::CalculateTarget(std::shared_ptr<TFDebug> tf_debug, const camera::CameraParams &params)
    : tf_debug_ptr_(std::move(tf_debug)), params_(params)
#endif
#ifndef RVIZ_DEBUG
CalculateTarget::CalculateTarget(const camera::CameraParams &params)
    : params_(params)
#endif
{
    receive_openmv_data_ = std::make_shared<ReceiveOpenMVData>();
}

cv::Point2d CalculateTarget::Decode_Openmv_Data(std::string &input_str) {
    double point_x = 0, point_y = 0;
    try {
        if (input_str == "null") {
            return {camera::NULL_ERROR, camera::NULL_ERROR};
        }
        nlohmann::json j = nlohmann::json::parse(input_str);
        point_x = j["cx"];
        point_y = j["cy"];
    } catch (const std::exception &e) {
        std::cerr << "C 发生错误" << e.what() << std::endl;
    }
    return {point_x, point_y};
}

cv::Point2d CalculateTarget::Transform_Image_TO_Real(cv::Point2d &image_point,
                                                     geometry_msgs::msg::TransformStamped pose) {
    double height = pose.transform.translation.z + params_.height_offset; // 无人机高度 + 相机安装高度
    double theta_x, theta_z;
    double length;
    double cam_pitch;
    double real_x, real_y;
    double w = pose.transform.rotation.w;
    double x = pose.transform.rotation.x;
    double y = pose.transform.rotation.y;
    double z = pose.transform.rotation.z;

    std::cerr << "w=" << w << "  x=" << x << "  y=" << y << "  z=" << z << std::endl;

    std::cerr << "X=" << pose.transform.translation.x << "  Y=" << pose.transform.translation.y << "  Z=" << pose.
            transform.translation.z << std::endl;

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

    R_DC = Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX()) // roll
           * Eigen::AngleAxisd(cam_pitch, Eigen::Vector3d::UnitY()) // pitch
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

cv::Point2d CalculateTarget::Handle_Openmv_Data(const geometry_msgs::msg::TransformStamped &pose) {
    try {
        while (true) {
            std::string input_str = receive_openmv_data_.get()->Receive_Openmv_Data();
            cv::Point2d temp_point = Decode_Openmv_Data(input_str);

            if (temp_point == cv::Point2d(camera::NULL_ERROR, camera::NULL_ERROR)) {
                //std::cout << temp_point.x << " " << temp_point.y << std::endl;
                return temp_point;
                continue;
            }
            cv::Point2d output_point = Transform_Image_TO_Real(temp_point, pose);

            //std::cout << output_point.x << " " << output_point.y << std::endl;
            return output_point;
        }
    } catch (const std::exception &e) {
        std::cerr << "D 发生错误" << e.what() << std::endl;
    }
    return {0, 0};
}

///tfdebug--------------------------------------------------------------------------------------------------------------------------
TFDebug::TFDebug(std::string parent, std::string child, double X, double Y, double Z)
    : child_(std::move(child)), parent_(std::move(parent)),
      X_(X), Y_(Y), Z_(Z) {
    node_ = rclcpp::Node::make_shared("tf_debug_");

    transform_ <<
            1, 0, 1,
            0, 1, 0,
            0, 0, 1;

    // 初始化TF广播器
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(node_);

    RCLCPP_INFO(node_->get_logger(), "TFDebug initialized: %s -> %s",
                parent_.c_str(), child_.c_str());
}

void TFDebug::broadcast_transform() {
    //RCLCPP_INFO(node_->get_logger(), "TFDebug running");

    geometry_msgs::msg::TransformStamped transform_msg;

    // 设置时间戳和坐标系
    transform_msg.header.stamp = node_->get_clock()->now();
    transform_msg.header.frame_id = parent_;
    transform_msg.child_frame_id = child_;

    // 设置平移分量
    transform_msg.transform.translation.x = X_;
    transform_msg.transform.translation.y = Y_;
    transform_msg.transform.translation.z = Z_;

    // 将Eigen旋转矩阵转换为四元数
    Eigen::Quaterniond eigen_quat(transform_);
    // 转换为geometry_msgs格式
    tf2::convert(eigen_quat, transform_msg.transform.rotation);

    // 广播变换
    tf_broadcaster_->sendTransform(transform_msg);
}

void TFDebug::SetTransform(Eigen::Matrix3d transform) {
    transform_ = std::move(transform);
}
