#ifndef CAMERA_PARAMS_H
#define CAMERA_PARAMS_H

#include <string>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <cmath>

// 工具宏：用于将 PROJECT_PATH 编译宏展开为字符串
#define CAM_STR(s) #s
#define CAM_MACRO_TO_STR(s) CAM_STR(s)

/// 默认 YAML 配置文件路径（编译时确定）
#define CAMERA_YAML_PATH CAM_MACRO_TO_STR(PROJECT_PATH) "/config/param.yaml"

namespace camera {

/// 写死的常量（不会变，不需要放 YAML）
constexpr int IMAGE_WIDTH = 640;
constexpr int IMAGE_HEIGHT = 480;
constexpr int NULL_ERROR = 641;         // 无检测时的哨兵值

/// 相机标定参数，从 YAML 配置文件读取（换相机/镜头时改 YAML 即可）
struct CameraParams {
    double horizon_x = 41.8;       // 水平半视场角（度）
    double horizon_z = 33.9;       // 垂直半视场角（度）
    double cam_pitch_deg = -60.0;  // 安装俯仰角（度）
    double height_offset = 0.15;   // 距无人机高度（米）
    int device_index = 2;          // /dev/videoX

    /// 从 YAML 文件加载指定相机的参数
    static CameraParams load(const std::string &yaml_path, const std::string &camera_name) {
        CameraParams p;
        try {
            YAML::Node config = YAML::LoadFile(yaml_path);
            auto cam = config[camera_name];
            if (!cam) {
                std::cerr << "[CameraParams] camera '" << camera_name
                          << "' not found in " << yaml_path << ", using defaults" << std::endl;
                return p;
            }
            p.horizon_x      = cam["horizon_x"].as<double>(p.horizon_x);
            p.horizon_z      = cam["horizon_z"].as<double>(p.horizon_z);
            p.cam_pitch_deg  = cam["cam_pitch_deg"].as<double>(p.cam_pitch_deg);
            p.height_offset  = cam["height_offset"].as<double>(p.height_offset);
            p.device_index   = cam["device_index"].as<int>(p.device_index);
        } catch (const std::exception &e) {
            std::cerr << "[CameraParams] failed to load from " << yaml_path
                      << ": " << e.what() << ", using defaults" << std::endl;
        }
        return p;
    }

    // 便捷方法：角度转弧度、图像中心
    [[nodiscard]] double horizon_x_rad() const { return horizon_x * M_PI / 180.0; }
    [[nodiscard]] double horizon_z_rad() const { return horizon_z * M_PI / 180.0; }
    [[nodiscard]] double cam_pitch_rad()  const { return cam_pitch_deg * M_PI / 180.0; }
    static constexpr double cx() { return IMAGE_WIDTH / 2.0; }
    static constexpr double cy() { return IMAGE_HEIGHT / 2.0; }
};

} // namespace camera

#endif // CAMERA_PARAMS_H
