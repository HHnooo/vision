#include "Camera_driver.hpp"

using namespace std;
using namespace cv;

usb_camera::USBCamera::USBCamera(int device_index)
    : exposure_(0),
      ios_(0),
      gain_(0),
      device_index_(device_index),
      If_OpenCameraDevice(false) {
}

bool usb_camera::USBCamera::OpenCameraDevice() {
    try {
        cap_.open(device_index_);
        If_OpenCameraDevice = true;
        return true;
    } catch (...) {
        cerr << "Open USB camera device failed." << endl;
        return false;
    }
}

Mat usb_camera::USBCamera::GetFrame() {
    if (If_OpenCameraDevice) {
        cap_.read(frame_);
        return frame_;
    }
    return {};
}

bool usb_camera::USBCamera::SetIndex(int index) {
    if (!If_OpenCameraDevice) {
        return false;
    }
    device_index_ = index;
    return true;
}


bool usb_camera::USBCamera::SetExposure(int exposure) {
    if (!If_OpenCameraDevice) {
        return false;
    }
    exposure_ = exposure;
    cap_.set(CAP_PROP_EXPOSURE, exposure);
    return true;
}

bool usb_camera::USBCamera::SetIOS(int ios) {
    if (!If_OpenCameraDevice) {
        return false;
    }
    ios_ = ios;
    cap_.set(CAP_PROP_BRIGHTNESS, ios);
    return true;
}

bool usb_camera::USBCamera::SetGain(int gain) {
    if (!If_OpenCameraDevice) {
        return false;
    }
    gain_ = gain;
    cap_.set(CAP_PROP_GAIN, gain);
    return true;
}

bool usb_camera::USBCamera::SetResolution(int width, int height) {
    if (!If_OpenCameraDevice) {
        return false;
    }
    cap_.set(CAP_PROP_FRAME_WIDTH, width);
    cap_.set(CAP_PROP_FRAME_HEIGHT, height);
    return true;
}

bool usb_camera::USBCamera::SetFPS(int fps) {
    if (!If_OpenCameraDevice) {
        return false;
    }
    cap_.set(CAP_PROP_FPS, fps);
    return true;
}