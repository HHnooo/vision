#ifndef CAMERA_DATA_HANDLE_KALMANBOX_H
#define CAMERA_DATA_HANDLE_KALMANBOX_H

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
namespace camera {
    class KalmanBoxTracker {
    public:
        explicit KalmanBoxTracker(const Rect& initBox) {
            // x,y,w,h,vx,vy,vw,vh    x,y,w,h
            kf_ = KalmanFilter(8, 4, 0);

            // 状态转移矩阵(F)
            kf_.transitionMatrix = (Mat_<float>(8, 8) <<
                1, 0, 0, 0, 1, 0, 0, 0,
                0, 1, 0, 0, 0, 1, 0, 0,
                0, 0, 1, 0, 0, 0, 1, 0,
                0, 0, 0, 1, 0, 0, 0, 1,
                0, 0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 1);

            // 测量矩阵  测量噪声的方差
            setIdentity(kf_.measurementMatrix);

            // 过程噪声协方差  模型噪声协方差矩阵
            setIdentity(kf_.processNoiseCov, Scalar::all(1e-3));

            // 测量噪声协方差
            //setIdentity(kf_.measurementNoiseCov, Scalar::all(1e-1));
            kf_.measurementNoiseCov = Mat::zeros(4, 4, CV_32F);
            kf_.measurementNoiseCov.at<float>(0, 0) = 1e-1;  // x坐标
            kf_.measurementNoiseCov.at<float>(1, 1) = 1e-1;  // y坐标
            kf_.measurementNoiseCov.at<float>(2, 2) = 1e-1;   // 宽度
            kf_.measurementNoiseCov.at<float>(3, 3) = 1e-1;   // 高度

            // 后验误差协方差
            setIdentity(kf_.errorCovPost, Scalar::all(1e-2));

            // 状态
            Point2f center = Point2f(static_cast<float>(initBox.x + initBox.width/2.0),
                                    static_cast<float>(initBox.y + initBox.height/2.0));
            kf_.statePost = (Mat_<float>(8, 1) <<
                center.x, center.y, initBox.width, initBox.height, 0, 0, 0, 0);
        }

        // 预测目标位置
        Point2f predict() {
            Mat prediction = kf_.predict();
            return Point2f(prediction.at<float>(0), prediction.at<float>(1));
        }

        void update(const Rect& measBox) {
            Point2f center = Point2f(static_cast<float>(measBox.x + measBox.width/2.0),
                                    static_cast<float>(measBox.y + measBox.height/2.0));
            Mat measurement = (Mat_<float>(4, 1) <<
                center.x, center.y, measBox.width, measBox.height);
            //Mat err = measurement - kf_.measurementMatrix * kf_.statePost;
            //Mat mahalanobis = err.t() * kf_.errorCovPost.inv() * err;
            // if (mahalanobis.at<float>(0) > 100.0) {
            //     float originalWNoise = kf_.measurementNoiseCov.at<float>(2, 2);
            //     float originalHNoise = kf_.measurementNoiseCov.at<float>(3, 3);
            //     kf_.measurementNoiseCov.at<float>(2, 2) *= 10.0;
            //     kf_.measurementNoiseCov.at<float>(3, 3) *= 10.0;
            //
            //     kf_.correct(measurement);
            //
            //     // 恢复原值
            //     kf_.measurementNoiseCov.at<float>(2, 2) = originalWNoise;
            //     kf_.measurementNoiseCov.at<float>(3, 3) = originalHNoise;
            // } else {
                kf_.correct(measurement);
            //}
        }

        Rect get_state() {
            Mat state = kf_.statePost;
            float w = state.at<float>(2);
            float h = state.at<float>(3);
            float x = state.at<float>(0) - w/2;
            float y = state.at<float>(1) - h/2;
            return Rect(x, y, w, h);
        }

    private:
        KalmanFilter kf_;
    };

}

#endif
