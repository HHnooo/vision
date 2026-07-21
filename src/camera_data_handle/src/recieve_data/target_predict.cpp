#include <utility>

#include "../../include/camera_data_handle/target_predict.h"
#include <cmath>
#include <algorithm>

DroneTargetingSystem::DroneTargetingSystem(double gravity,
                                           double accuracy_threshold,
                                           size_t max_history_size,
                                           double recent_data_weight)
    : gravity_(gravity),
      accuracy_threshold_(accuracy_threshold),
      max_history_size_(max_history_size),
      recent_data_weight_(recent_data_weight),
      estimated_target_velocity_(0, 0),
      velocity_confidence_(0.0) {
}

void DroneTargetingSystem::setDroneState(const Vector2D &position,
                                         double height,
                                         const Vector2D &velocity) {
    drone_position_ = position;
    drone_height_ = height;
    drone_velocity_ = velocity;
    last_drone_update_ = std::chrono::system_clock::now();
}

void DroneTargetingSystem::addTargetObservation(const Vector2D &position,
                                                std::chrono::system_clock::time_point timestamp,
                                                double confidence) {
    confidence = std::max(0.0, std::min(1.0, confidence));

    target_observations_.emplace_back(position, timestamp, confidence);

    std::sort(target_observations_.begin(), target_observations_.end(),
              [](const TargetObservation &a, const TargetObservation &b) {
                  return a.timestamp < b.timestamp;
              });

    if (target_observations_.size() > max_history_size_) {
        target_observations_.erase(target_observations_.begin());
    }

    estimateTargetVelocity();
}

double DroneTargetingSystem::getTimeDifference(const std::chrono::system_clock::time_point &t1,
                                               const std::chrono::system_clock::time_point &t2) {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    return duration.count() / 1000.0;
}

void DroneTargetingSystem::estimateTargetVelocity() {
    if (target_observations_.size() < 2) {
        velocity_confidence_ = 0.0;
        estimated_target_velocity_ = Vector2D(0, 0);
        return;
    }

    std::vector<std::pair<Vector2D, double> > velocity_estimates;
    double total_weight = 0.0;

    for (size_t i = 1; i < target_observations_.size(); ++i) {
        const auto &prev = target_observations_[i - 1];
        const auto &curr = target_observations_[i];

        double dt = getTimeDifference(prev.timestamp, curr.timestamp);
        if (dt <= 0) continue; // 防止时间戳错误

        Vector2D vel(
            (curr.position.x - prev.position.x) / dt,
            (curr.position.y - prev.position.y) / dt
        );

        double time_weight = 1.0 / (1.0 + dt); // 时间差小的权重高
        double conf_weight = (prev.confidence + curr.confidence) / 2.0; // 平均置信度
        double recency_weight = pow(recent_data_weight_, target_observations_.size() - i); // 数据权重

        double weight = time_weight * conf_weight * recency_weight;
        velocity_estimates.emplace_back(vel, weight);
        total_weight += weight;
    }

    if (total_weight > 0 && !velocity_estimates.empty()) {
        Vector2D avg_vel(0, 0);
        for (const auto &[vel, weight]: velocity_estimates) {
            avg_vel = avg_vel + vel * (weight / total_weight);
        }
        estimated_target_velocity_ = avg_vel;

        double consistency = 1.0;
        if (velocity_estimates.size() > 1) {
            double vel_var = 0.0;
            for (const auto &[vel, weight]: velocity_estimates) {
                double diff = distance(vel, avg_vel);
                vel_var += diff * diff * (weight / total_weight);
            }
            consistency = 1.0 / (1.0 + sqrt(vel_var));
        }

        velocity_confidence_ = consistency *
                               (1.0 - exp(-0.5 * target_observations_.size()));
    } else {
        velocity_confidence_ = 0.0;
        estimated_target_velocity_ = Vector2D(0, 0);
    }
}

double DroneTargetingSystem::calculateDropTime() const {
    if (drone_height_ <= 0) return 0.0;
    return sqrt(2 * drone_height_ / gravity_);
}

Vector2D DroneTargetingSystem::predictTargetPosition(double t) const {
    if (target_observations_.empty()) {
        return Vector2D(0, 0);
    }

    const auto &last_obs = target_observations_.back();
    return Vector2D(
        last_obs.position.x + estimated_target_velocity_.x * t,
        last_obs.position.y + estimated_target_velocity_.y * t
    );
}

Vector2D DroneTargetingSystem::calculateImpactPosition() const {
    double drop_time = calculateDropTime();

    return Vector2D(
        drone_position_.x + drone_velocity_.x * drop_time,
        drone_position_.y + drone_velocity_.y * drop_time
    );
}

bool DroneTargetingSystem::getPrediction(Vector2D &impact_pos,
                                         Vector2D &target_pos,
                                         double &drop_time) const {
    if (target_observations_.size() < 2 || velocity_confidence_ < 0.3) {
        return false;
    }

    drop_time = calculateDropTime();
    impact_pos = calculateImpactPosition();
    target_pos = predictTargetPosition(drop_time);
    return true;
}

bool DroneTargetingSystem::shouldDropItem() const {
    if (target_observations_.size() < 3 || velocity_confidence_ < 0.5) {
        return false;
    }

    double drop_time = calculateDropTime();
    Vector2D impact_pos = calculateImpactPosition();
    Vector2D target_pos = predictTargetPosition(drop_time);

    double error = distance(impact_pos, target_pos);
    return error < accuracy_threshold_;
}

double DroneTargetingSystem::distance(const Vector2D &a, const Vector2D &b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}
