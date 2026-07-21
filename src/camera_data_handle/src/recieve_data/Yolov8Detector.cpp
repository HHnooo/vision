#include "../../include/Yolov8Detector/Yolov8Detector.h"

#include<CL/opencl.hpp>
#include <CL/opencl.h>
#include <iostream>
#include <rclcpp/logging.hpp>
#include <openvino/openvino.hpp>
#include <stdexcept>
#include <fstream>

using namespace cv;
using namespace dnn;
using namespace std;

#ifdef RUN_THE_FUNCTION_BY_GPU

bool Yolov8::YOLOv8Detector::initOpenCL() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() == 0) {
        cerr << "No platforms found" << endl;
        return false;
    }

    bool if_find_gpu = false;
    for (const auto &platform: platforms) {
        if (platform.getInfo<CL_PLATFORM_VENDOR>().find("Intel") != std::string::npos) {
            opencl_platform_ = platform;
            if_find_gpu = true;
            break;
        }
    }

    if (!if_find_gpu) {
        opencl_platform_ = platforms[0];
        std::cout << "Using platform: " << opencl_platform_.getInfo<CL_PLATFORM_VENDOR>() << std::endl;
    }

    std::vector<cl::Device> devices;
    opencl_platform_.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.size() == 0) {
        cerr << "No GPU devices found" << endl;
        opencl_platform_.getDevices(CL_DEVICE_TYPE_CPU, &devices);
    }

    if (devices.size() == 0) {
        cerr << "No devices found" << endl;
        return false;
    }

    opencl_device_ = devices[0];
    std::cout << "Using device: " << opencl_device_.getInfo<CL_DEVICE_NAME>() << std::endl;

    opencl_context_ = cl::Context(opencl_device_);
    opencl_command_queue_ = cl::CommandQueue(opencl_context_);

    std::string kernelSource = loadKernelSource("../kernal_msg/function.cl");

    const char *source_str = kernelSource.data();
    size_t source_size = kernelSource.length();
    cl::Program::Sources sources(1, kernelSource);
    opencl_program_ = cl::Program(opencl_context_, sources);

    cl_int err = opencl_program_.build({opencl_device_});
    if (err != CL_SUCCESS) {
        std::cerr << "Kernel build error: " << opencl_program_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(opencl_device_) <<
                std::endl;
        return false;
    }

    CHECK_ERROR(err);

    score_kernel_ = cl::Kernel(opencl_program_, "process_scores", &err);

    CHECK_ERROR(err);

    nms_kernel_ = cl::Kernel(opencl_program_, "nms", &err);

    CHECK_ERROR(err);

    return true;
}

std::string Yolov8::YOLOv8Detector::loadKernelSource(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open kernel file: " + filename);
    }

    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}
#endif

Yolov8::YOLOv8Detector::YOLOv8Detector(const string &modelPath, const vector<string> &classNames, int imgSize,
                                       float confThreshold, float nmsThreshold, int imgWidth, int imgHeight)
    : classNames_(classNames), imgSize_(imgSize),
      confThreshold_(confThreshold), nmsThreshold_(nmsThreshold),
      imgHeight_(imgHeight), imgWidth_(imgWidth) {
#ifndef DETECTION_OPENVINO_OPEN
    net_ = readNet(modelPath);
    net_.setPreferableBackend(DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(DNN_TARGET_CPU);
#endif

#ifdef DETECTION_OPENVINO_OPEN
    try {
        auto model = core_.read_model(modelPath);
        auto inputs = model->inputs();
        // 新模型可能没有tensor名称，改用索引方式，不存名字
        auto in_names = inputs[0].get_names();
        inputName_ = in_names.empty() ? "" : *in_names.begin();
        inputCount_ = inputs.size();

        auto outputs = model->outputs();
        outputCount_ = outputs.size();
        for (const auto &output: outputs) {
            auto out_names = output.get_names();
            outputNames_.push_back(out_names.empty() ? "" : *out_names.begin());
        }

        ov::Shape inputShape = inputs[0].get_shape();
        inputShape[0] = 1;
        inputShape[2] = imgSize_;
        inputShape[3] = imgSize_;
        model->reshape({inputShape});

        for (const auto &device: core_.get_available_devices()) {
            std::cout << "Available device: " << device << std::endl;
        }

        ov::AnyMap config;
        config["GPU_DISABLE_WINOGRAD_CONVOLUTION"] = "True";

        //compiled_model_ = core_.compile_model(model, "GPU", config);
        compiled_model_ = core_.compile_model(model, "CPU");

        infer_request_ = compiled_model_.create_infer_request();
    } catch (const std::exception &e) {
        std::cerr << "cannot initialize : " << e.what() << std::endl;
        throw;
    }
#endif
}

vector<Yolov8::Detection> Yolov8::YOLOv8Detector::detect(Mat &image) {
    Mat blob = preprocess(image);
#ifndef DETECTION_OPENVINO_OPEN

    net_.setInput(blob);

    vector<Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());

    return postprocess(outputs, image.size());
#endif

#ifdef DETECTION_OPENVINO_OPEN

    ov::Shape temp_shape;
    temp_shape.push_back(1);
    temp_shape.push_back(3);
    temp_shape.push_back(imgSize_);
    temp_shape.push_back(imgSize_);

    ov::Tensor inputTensor(ov::element::f32,
                           temp_shape,
                           blob.ptr<float>());

    // 优先用索引，名字为空时也能工作
    if (!inputName_.empty())
        infer_request_.set_tensor(inputName_, inputTensor);
    else
        infer_request_.set_input_tensor(0, inputTensor);

    infer_request_.infer();

    std::vector<cv::Mat> outputs;
    for (size_t i = 0; i < outputCount_; i++) {
        ov::Tensor outputTensor = outputNames_[i].empty()
            ? infer_request_.get_output_tensor(i)
            : infer_request_.get_tensor(outputNames_[i]);
        const float *data = outputTensor.data<float>();
        ov::Shape shape = outputTensor.get_shape();

        int rows = shape[1];
        int cols = shape[2];
        cv::Mat output(cols, rows, CV_32F);
        memcpy(output.data, data, rows * cols * sizeof(float));
        outputs.push_back(output);
    }

    return postprocess(outputs, image.size());

#endif
}

Mat Yolov8::YOLOv8Detector::preprocess(const Mat &image) const {
    double scale = 1.0 / 255.0;
    Scalar mean = Scalar(0, 0, 0);
    bool swapRB = true;

    Mat blob;
    blobFromImage(image, blob, scale, Size(imgSize_, imgSize_),
                  mean, swapRB, false, CV_32F);
    return blob;
}


vector<Yolov8::Detection> Yolov8::YOLOv8Detector::postprocess(const vector<Mat> &outputs, const Size &originalSize) {
    vector<Detection> detections;

#ifdef DETECTION_OPENVINO_OPEN
#ifdef RUN_THE_FUNCTION_BY_GPU
    if (outputs.empty()) {
        return detections;
    }

    cv::Mat output = outputs[0];
    output = output.reshape(1, output.size[1]);
    cv::transpose(output, output);

    int origW = originalSize.width;
    int origH = originalSize.height;
    int numDetections = output.rows;
    int numClasses = classNames_.size();

    cl::Buffer d_output(opencl_context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        output.total() * sizeof(float), output.data);

    std::vector<int> classIds(numDetections, -1);
    std::vector<float> confidences(numDetections, 0.0f);
    std::vector<float> boxes(numDetections * 4, 0.0f); // x, y, width, height
    std::vector<int> indices(numDetections, 0);

    cl::Buffer d_classIds(opencl_context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          classIds.size() * sizeof(int), classIds.data());
    cl::Buffer d_confidences(opencl_context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                             confidences.size() * sizeof(float), confidences.data());
    cl::Buffer d_boxes(opencl_context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                       boxes.size() * sizeof(float), boxes.data());
    cl::Buffer d_indices(opencl_context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                         indices.size() * sizeof(int), indices.data());

    score_kernel_.setArg(0, d_output);
    score_kernel_.setArg(1, numDetections);
    score_kernel_.setArg(2, numClasses);
    score_kernel_.setArg(3, confThreshold_);
    score_kernel_.setArg(4, origW);
    score_kernel_.setArg(5, origH);
    score_kernel_.setArg(6, imgSize_);
    score_kernel_.setArg(7, d_classIds);
    score_kernel_.setArg(8, d_confidences);
    score_kernel_.setArg(9, d_boxes);

    size_t globalWorkSize = numDetections;
    size_t localWorkSize = 1024; // 可以根据设备调整
    if (globalWorkSize % localWorkSize != 0) {
        globalWorkSize += localWorkSize - (globalWorkSize % localWorkSize);
    }

    opencl_command_queue_.enqueueNDRangeKernel(score_kernel_, cl::NullRange, cl::NDRange(globalWorkSize),
                                               cl::NDRange(localWorkSize));

    opencl_command_queue_.enqueueReadBuffer(d_classIds, CL_TRUE, 0, classIds.size() * sizeof(int), classIds.data());
    int validCount = 0;
    for (int id: classIds) {
        if (id != -1) validCount++;
    }

    if (validCount == 0) {
        return detections;
    }

    nms_kernel_.setArg(0, d_boxes);
    nms_kernel_.setArg(1, d_confidences);
    nms_kernel_.setArg(2, numDetections);
    nms_kernel_.setArg(3, nmsThreshold_);
    nms_kernel_.setArg(4, d_indices);

    opencl_command_queue_.enqueueNDRangeKernel(nms_kernel_, cl::NullRange, cl::NDRange(1), cl::NullRange);

    opencl_command_queue_.enqueueReadBuffer(d_indices, CL_TRUE, 0, indices.size() * sizeof(int), indices.data());
    opencl_command_queue_.enqueueReadBuffer(d_classIds, CL_TRUE, 0, classIds.size() * sizeof(int), classIds.data());
    opencl_command_queue_.enqueueReadBuffer(d_confidences, CL_TRUE, 0, confidences.size() * sizeof(float),
                                            confidences.data());
    opencl_command_queue_.enqueueReadBuffer(d_boxes, CL_TRUE, 0, boxes.size() * sizeof(float), boxes.data());

    for (int i = 0; i < validCount; i++) {
        int idx = indices[i];
        if (idx < 0 || idx >= numDetections) break;

        Detection detection;
        detection.classId = classIds[idx];
        detection.confidence = confidences[idx];
        detection.box = cv::Rect(
            static_cast<int>(boxes[idx * 4]), // x
            static_cast<int>(boxes[idx * 4 + 1]), // y
            static_cast<int>(boxes[idx * 4 + 2]), // width
            static_cast<int>(boxes[idx * 4 + 3]) // height
        );

        detections.push_back(detection);
    }

#endif


#ifndef RUN_THE_FUNCTION_BY_GPU
    Mat output = outputs[0];

    output = output.reshape(1, output.size[1]);
    transpose(output, output);

    int origW = originalSize.width;
    int origH = originalSize.height;

    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (int i = 0; i < output.rows; i++) {
        Mat scores = output.row(i).colRange(4, 4 + classNames_.size());

        Point classIdPoint;
        double maxScore;
        minMaxLoc(scores, 0, &maxScore, 0, &classIdPoint);

        if (maxScore > confThreshold_) {
            float cx = output.at<float>(i, 0);
            float cy = output.at<float>(i, 1);
            float w = output.at<float>(i, 2);
            float h = output.at<float>(i, 3);

            int left = static_cast<int>((cx - w / 2) * origW / imgSize_);
            int top = static_cast<int>((cy - h / 2) * origH / imgSize_);
            int width = static_cast<int>(w * origW / imgSize_);
            int height = static_cast<int>(h * origH / imgSize_);

            left = max(0, min(left, origW - 1));
            top = max(0, min(top, origH - 1));
            width = max(1, min(width, origW - left));
            height = max(1, min(height, origH - top));

            classIds.push_back(classIdPoint.x);
            confidences.push_back(static_cast<float>(maxScore));
            boxes.push_back(Rect(left, top, width, height));
        }
    }

    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold_, nmsThreshold_, indices); // 非极大值抑制

    for (int idx: indices) {
        Detection detection;
        detection.classId = classIds[idx];
        detection.confidence = confidences[idx];
        detection.box = boxes[idx];
        detections.push_back(detection);
    }
#endif
#endif
    return detections;
}

std::vector<std::vector<double> > Yolov8::YOLOv8Detector::drawDetections(
    Mat &image, const vector<Detection> &detections) {
    std::vector<std::vector<double> > position_container;
    for (const auto &detection: detections) {
        std::vector<double> position;

#ifndef NO_IMAGE
        rectangle(image, detection.box, Scalar(0, 255, 0), 2);
        string label = format("%s: %.2f", classNames_[detection.classId].c_str(), detection.confidence);

        int baseLine;

        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        Rect labelRect = Rect(detection.box.x, detection.box.y - labelSize.height - baseLine,
                              labelSize.width, labelSize.height + baseLine);

        rectangle(image, labelRect, Scalar(0, 255, 0), FILLED);
#endif

        // position.push_back(detection.box.x + detection.box.width / 2 - imgWidth_ / 2);
        // position.push_back(imgHeight_ / 2 - detection.box.y - detection.box.height / 2);

        position.push_back(detection.box.x + detection.box.width / 2 - imgWidth_ / 2);
        position.push_back(detection.box.y + detection.box.height / 2 - imgHeight_ / 2);

        //std::cout << detection.box.x + detection.box.width / 2 - imgWidth / 2 << endl;
        //std::cout << imgHeight / 2 - detection.box.y - detection.box.height / 2 << endl;
#ifndef NO_IMAGE
        circle(image, Point(detection.box.x + detection.box.width / 2, detection.box.y + detection.box.height / 2), 3,
               Scalar(0, 0, 255), FILLED);

        putText(image, label, Point(detection.box.x, detection.box.y - baseLine),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
#endif

        position_container.push_back(position);
    }
    return position_container;
}
