#ifndef CAMERA_DATA_HANDLE_TARGET_PREDICT_H
#define CAMERA_DATA_HANDLE_TARGET_PREDICT_H

#ifndef DRONE_TARGETING_SYSTEM_H
#define DRONE_TARGETING_SYSTEM_H

#include <vector>
#include <chrono>

struct Vector2D {
    double x;
    double y;

    Vector2D(double x = 0, double y = 0) : x(x), y(y) {
    }

    Vector2D operator+(const Vector2D &other) const {
        return {x + other.x, y + other.y};
    }

    Vector2D operator-(const Vector2D &other) const {
        return {x - other.x, y - other.y};
    }

    Vector2D operator*(double scalar) const {
        return {x * scalar, y * scalar};
    }
};

struct TargetObservation {
    Vector2D position;
    std::chrono::system_clock::time_point timestamp;
    double confidence;

    TargetObservation(Vector2D pos, std::chrono::system_clock::time_point ts, double conf = 1.0)
        : position(pos), timestamp(ts), confidence(conf) {
    }
};

class DroneTargetingSystem {
private:

    double gravity_; // 重力加速度
    double accuracy_threshold_; // 投掷精度

    Vector2D drone_position_;
    Vector2D drone_velocity_;
    double drone_height_;
    std::chrono::system_clock::time_point last_drone_update_;

    std::vector<TargetObservation> target_observations_;
    size_t max_history_size_;
    double recent_data_weight_;

    Vector2D estimated_target_velocity_;
    double velocity_confidence_;

    double getTimeDifference(const std::chrono::system_clock::time_point &t1,
                             const std::chrono::system_clock::time_point &t2);

    void estimateTargetVelocity();

public:

    explicit DroneTargetingSystem(double gravity = 9.81,
                         double accuracy_threshold = 1.5,
                         size_t max_history_size = 10,
                         double recent_data_weight = 0.8);

    void setDroneState(const Vector2D &position, double height, const Vector2D &velocity);

    void addTargetObservation(const Vector2D &position,
                              std::chrono::system_clock::time_point timestamp = std::chrono::system_clock::now(),
                              double confidence = 1.0);

    double calculateDropTime() const;

    Vector2D predictTargetPosition(double t) const;

    Vector2D calculateImpactPosition() const;

    bool getPrediction(Vector2D &impact_pos, Vector2D &target_pos, double &drop_time) const;

    bool shouldDropItem() const;

    double getVelocityConfidence() const { return velocity_confidence_; }

    static double distance(const Vector2D &a, const Vector2D &b);
};

#endif // DRONE_TARGETING_SYSTEM_H


#endif //CAMERA_DATA_HANDLE_TARGET_PREDICT_H
