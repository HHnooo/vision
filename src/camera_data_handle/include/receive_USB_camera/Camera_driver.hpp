#ifndef CAMERA_DATA_HANDLE_CAMERA_DRIVER_HPP
#define CAMERA_DATA_HANDLE_CAMERA_DRIVER_HPP

#include <opencv4/opencv2/opencv.hpp>

namespace usb_camera {
    class USBCamera {
    public:
        explicit USBCamera(int device_index=2);

        bool OpenCameraDevice();

        cv::Mat GetFrame();

        bool SetIndex(int index);

        bool SetExposure(int exposure);

        bool SetIOS(int ios);

        bool SetGain(int gain);

        bool SetResolution(int width, int height);

        bool SetFPS(int fps);
    private:
        cv::Mat frame_{};
        cv::VideoCapture cap_;
        int exposure_;
        int ios_;
        int gain_;
        int device_index_;
        bool If_OpenCameraDevice;
    };
}


#endif //CAMERA_DATA_HANDLE_CAMERA_DRIVER_HPP