#ifndef YOLOV8DETECTOR_H
#define YOLOV8DETECTOR_H

#define CHECK_ERROR(err) \
if(err!=CL_SUCCESS){\
std::cerr<<"OpenCL error at line"<<__LINE__<<"."<<err<<std::endl;\
exit(EXIT_FAILURE);\
}

#include <CL/cl2.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <openvino/openvino.hpp>

namespace Yolov8 {
    inline std::vector<std::string> defaultClassNames = {
        "H", "tent", "car",
        "bridge", "pillbox", "tank", "Red_cross"
    };

    struct Detection {
        int classId;
        float confidence;
        cv::Rect box;
        int class_id{};

#ifdef KEY_POINT_TRACKING
        std::vector<cv::Point2f> features;
#endif
    };

    class YOLOv8Detector {
    public:
        virtual ~YOLOv8Detector() = default;

        explicit YOLOv8Detector(const std::string &modelPath,
                                const std::vector<std::string> &classNames = defaultClassNames,
                                int imgSize = 160,
                                float confThreshold = 0.7,
                                float nmsThreshold = 0.4,
                                int imgWidth = 640,
                                int imgHeigh = 480
        );

        void setConfidenceThreshold(float threshold) { confThreshold_ = threshold; }

        void setNmsThreshold(float threshold) { nmsThreshold_ = threshold; }

        virtual std::vector<Detection> detect(cv::Mat &image);

        std::vector<std::vector<double> > drawDetections(cv::Mat &image, const std::vector<Detection> &detections);

    private:
#ifdef RUN_THE_FUNCTION_BY_GPU

        cl::Platform opencl_platform_;
        cl::Program opencl_program_;
        cl::Device opencl_device_;
        cl::Context opencl_context_;
        cl::CommandQueue opencl_command_queue_;
        cl::Kernel score_kernel_;
        cl::Kernel nms_kernel_;

        cl::Buffer d_output_;
        cl::Buffer d_classIds_;
        cl::Buffer d_confidences_;
        cl::Buffer d_boxes_;
        cl::Buffer d_indices_;

        bool initOpenCL();

        static std::string loadKernelSource(const std::string &filename);

#endif

#ifdef DETECTION_OPENVINO_OPEN
        ov::Core core_{};
        ov::CompiledModel compiled_model_{};
        ov::InferRequest infer_request_{};
        ov::Shape input_shape_{};
        std::shared_ptr<ov::Model> model_{};
        std::string inputName_{};
        std::vector<std::string> outputNames_;
        size_t inputCount_{0};
        size_t outputCount_{0};
#endif


#ifndef DETECTION_OPENVINO_OPEN
        cv::dnn::Net net_;
#endif

        std::vector<std::string> classNames_;
        int imgSize_{320};
        float confThreshold_;
        float nmsThreshold_;
        int imgWidth_;
        int imgHeight_;

        [[nodiscard]] cv::Mat preprocess(const cv::Mat &image) const;

        std::vector<Detection> postprocess(const std::vector<cv::Mat> &outputs, const cv::Size &originalSize);
    };
}

#endif
