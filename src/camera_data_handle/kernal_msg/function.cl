// 处理分数计算和边界框初步筛选
__kernel void process_scores(
    __global const float* output,
    const int numDetections,
    const int numClasses,
    const float confThreshold,
    const int origW,
    const int origH,
    const int imgSize,
    __global int* classIds,
    __global float* confidences,
    __global float* boxes) {

    int i = get_global_id(0);
    if (i >= numDetections) return;

    // 找到最大分数和对应的类别ID
    float maxScore = 0.0f;
    int classId = -1;

    for (int c = 0; c < numClasses; c++) {
        float score = output[i * (numClasses + 4) + 4 + c];
        if (score > maxScore) {
            maxScore = score;
            classId = c;
        }
    }

    // 如果分数低于阈值，标记为无效
    if (maxScore <= confThreshold) {
        classIds[i] = -1;
        return;
    }

    // 计算边界框
    float cx = output[i * (numClasses + 4)];
    float cy = output[i * (numClasses + 4) + 1];
    float w = output[i * (numClasses + 4) + 2];
    float h = output[i * (numClasses + 4) + 3];

    // 转换到原始图像尺寸
    float left = (cx - w / 2.0f) * origW / imgSize;
    float top = (cy - h / 2.0f) * origH / imgSize;
    float width = w * origW / imgSize;
    float height = h * origH / imgSize;

    // 边界检查
    left = max(0.0f, min(left, origW - 1.0f));
    top = max(0.0f, min(top, origH - 1.0f));
    width = max(1.0f, min(width, origW - left));
    height = max(1.0f, min(height, origH - top));

    // 保存结果
    classIds[i] = classId;
    confidences[i] = maxScore;
    boxes[i * 4] = left;
    boxes[i * 4 + 1] = top;
    boxes[i * 4 + 2] = width;
    boxes[i * 4 + 3] = height;
}

// 非极大值抑制
__kernel void nms(
    __global const float* boxes,
    __global const float* confidences,
    const int numDetections,
    const float nmsThreshold,
    __global int* indices) {

    // 初始化索引
    for (int i = 0; i < numDetections; i++) {
        indices[i] = -1;  // -1表示无效
    }

    // 收集有效检测
    __global int* validIndices = (__global int*)malloc(numDetections * sizeof(int));
    int validCount = 0;

    for (int i = 0; i < numDetections; i++) {
        // 检查这个检测是否有效（在process_scores中标记）
        // 这里假设只有当confidences[i] > 0时才有效
        if (confidences[i] > 0.0f) {
            validIndices[validCount++] = i;
        }
    }

    // 按置信度排序（从高到低）
    for (int i = 0; i < validCount; i++) {
        for (int j = i + 1; j < validCount; j++) {
            if (confidences[validIndices[i]] < confidences[validIndices[j]]) {
                int temp = validIndices[i];
                validIndices[i] = validIndices[j];
                validIndices[j] = temp;
            }
        }
    }

    // 执行NMS
    int resultCount = 0;
    __global bool* suppressed = (__global bool*)malloc(numDetections * sizeof(bool));
    for (int i = 0; i < numDetections; i++) {
        suppressed[i] = false;
    }

    for (int i = 0; i < validCount; i++) {
        int idx = validIndices[i];
        if (suppressed[idx]) continue;

        indices[resultCount++] = idx;

        // 计算与其他框的IOU并抑制重叠度过高的框
        float x1 = boxes[idx * 4];
        float y1 = boxes[idx * 4 + 1];
        float w1 = boxes[idx * 4 + 2];
        float h1 = boxes[idx * 4 + 3];
        float area1 = w1 * h1;

        for (int j = i + 1; j < validCount; j++) {
            int idx2 = validIndices[j];
            if (suppressed[idx2]) continue;

            float x2 = boxes[idx2 * 4];
            float y2 = boxes[idx2 * 4 + 1];
            float w2 = boxes[idx2 * 4 + 2];
            float h2 = boxes[idx2 * 4 + 3];
            float area2 = w2 * h2;

            // 计算交叠区域
            float x1_ = max(x1, x2);
            float y1_ = max(y1, y2);
            float x2_ = min(x1 + w1, x2 + w2);
            float y2_ = min(y1 + h1, y2 + h2);

            float overlapWidth = max(0.0f, x2_ - x1_);
            float overlapHeight = max(0.0f, y2_ - y1_);
            float overlapArea = overlapWidth * overlapHeight;

            // 计算IOU
            float iou = overlapArea / (area1 + area2 - overlapArea);

            if (iou > nmsThreshold) {
                suppressed[idx2] = true;
            }
        }
    }

    free(validIndices);
    free(suppressed);
}
