#ifndef CAMERA_DATA_HANDLE_POINT_TRACKING_H
#define CAMERA_DATA_HANDLE_POINT_TRACKING_H

#include "../Yolov8Detector/Yolov8Detector.h"

#ifdef KEY_POINT_TRACKING
namespace Yolov8 {
    class ObjectTracker {
    public:
        explicit ObjectTracker(const cv::Ptr<cv::Feature2D> &detector = cv::ORB::create(),
                      const cv::Ptr<cv::DescriptorMatcher> &matcher = cv::DescriptorMatcher::create(
                          "BruteForce-Hamming"))
            : tracked_object_(), feature_detector_(detector), descriptor_matcher_(matcher) {
        }

        void init(const cv::Mat &image, const Yolov8::Detection &detection) {
            tracked_object_ = detection;

            //提取特征点
            cv::Mat roi = image(detection.box);

            std::vector<cv::KeyPoint> keypoints;

            // for (const auto& point : tracked_object_.features) {
            //     keypoints.emplace_back(point.x,point.y,1.0f);
            // }
            tracked_object_.features.emplace_back(tracked_object_.box.x,tracked_object_.box.y);
            keypoints.emplace_back(tracked_object_.box.x,tracked_object_.box.y,5.0f);

            feature_detector_->detectAndCompute(roi, cv::noArray(), keypoints, tracked_descriptors_);

            for (const auto& point:keypoints) {
                tracked_object_.features.emplace_back(point.pt.x, point.pt.y);
            }
        }

        bool update(const cv::Mat &image, Yolov8::Detection &result) {
            if (tracked_object_.features.empty()) return false;

            std::vector<cv::KeyPoint> frame_keypoints;
            cv::Mat frame_descriptors;
            feature_detector_->detectAndCompute(image, cv::noArray(), frame_keypoints, frame_descriptors);

            std::vector<cv::DMatch> matches;
            descriptor_matcher_->match(tracked_descriptors_, frame_descriptors, matches);

            std::sort(matches.begin(), matches.end());
            const int num_good_matches = matches.size() * 0.05;
            matches.erase(matches.begin() + num_good_matches, matches.end());

            if (matches.size() < 4) return false; // 至少需要4个点计算变换

            std::vector<cv::Point2f> obj_pts, frame_pts;
            for (const auto &m: matches) {
                obj_pts.push_back(tracked_object_.features[m.queryIdx]);
                frame_pts.push_back(frame_keypoints[m.trainIdx].pt);
            }

            cv::Mat transform = cv::estimateAffinePartial2D(obj_pts, frame_pts);
            if (transform.empty()) return false;

            std::vector<cv::Point2f> corners = {
                cv::Point2f(tracked_object_.box.x, tracked_object_.box.y),
                cv::Point2f(tracked_object_.box.x + tracked_object_.box.width, tracked_object_.box.y),
                cv::Point2f(tracked_object_.box.x + tracked_object_.box.width,
                            tracked_object_.box.y + tracked_object_.box.height),
                cv::Point2f(tracked_object_.box.x, tracked_object_.box.y + tracked_object_.box.height)
            };

            std::vector<cv::Point2f> transformed_corners;
            cv::transform(corners, transformed_corners, transform);

            cv::Rect new_box = cv::boundingRect(transformed_corners);
            result = tracked_object_;
            result.box = new_box;
            return true;
        }

    private:
        Detection tracked_object_;
        cv::Mat tracked_descriptors_;
        cv::Ptr<cv::Feature2D> feature_detector_;
        cv::Ptr<cv::DescriptorMatcher> descriptor_matcher_;
    };



    class YOLOv8Tracker : public Yolov8::YOLOv8Detector {
    public:
        using YOLOv8Detector::YOLOv8Detector;

        std::vector<Detection> detect(cv::Mat &image) override {

            auto detections = YOLOv8Detector::detect(image);
            active_detections_ = detections;

            // 初始化新目标的跟踪器
            for (auto &det: detections) {
                if (trackers_.find(det.class_id) == trackers_.end()) {
                    trackers_[det.class_id] = std::make_unique<ObjectTracker>();
                    trackers_[det.class_id]->init(image, det);
                }
            }

            return detections;
        }

        std::vector<Yolov8::Detection> track(const cv::Mat &image) {
            std::vector<Yolov8::Detection> results;

            for (auto &kv: trackers_) {
                Yolov8::Detection tracked_det;
                if (kv.second->update(image, tracked_det)) {
                    results.push_back(tracked_det);
                }
            }

            return results;
        }

    private:
        std::unordered_map<int, std::unique_ptr<ObjectTracker> > trackers_;
        std::vector<Yolov8::Detection> active_detections_;
    };
}

#endif

#endif //CAMERA_DATA_HANDLE_POINT_TRACKING_H
