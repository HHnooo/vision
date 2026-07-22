"""
OpenVINO 摄像头实时检测 — 适配 Intel NUC

仅需 openvino + opencv + numpy，不需要 ultralytics。
"""
import cv2
import numpy as np
import openvino as ov

# ============================================================
# 配置
# ============================================================
MODEL_XML = r"model_openvino\best.xml"   # NUC 上改为你的路径
CAM_INDEX = 0
CONF_THRES = 0.5
IOU_THRES = 0.45
IMGSZ = 640

class_names = {
    0: "H", 1: "tent", 2: "car", 3: "bridge",
    4: "pillbox", 5: "tank", 6: "Red_cross",
}
colors = [
    (0, 255, 255), (0, 165, 255), (0, 255, 0),
    (255, 0, 0), (255, 0, 255), (0, 200, 200), (255, 255, 255),
]


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(h * r), int(w * r)
    dw = (new_shape - nw) // 2
    dh = (new_shape - nh) // 2
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, dh, new_shape - nh - dh,
                             dw, new_shape - nw - dw,
                             cv2.BORDER_CONSTANT, value=color)
    return img, (r, dw, dh)


def preprocess(frame):
    img, pad = letterbox(frame, IMGSZ)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)[np.newaxis, ...]
    return img.astype(np.float32) / 255.0, pad


def postprocess(output, pad, frame_shape):
    r, dw, dh = pad
    fh, fw = frame_shape[:2]

    # YOLOv8 output: (1, 11, 8400) → 11 = 4(xywh) + 7(classes)
    output = output[0].T  # (8400, 11)
    boxes_raw = output[:, :4]
    scores_raw = output[:, 4:]

    class_ids = scores_raw.argmax(axis=1)
    scores = scores_raw.max(axis=1)

    mask = scores > CONF_THRES
    boxes_raw = boxes_raw[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(boxes_raw) == 0:
        return [], [], []

    # xywh → xyxy (640x640)
    xyxy = np.zeros_like(boxes_raw)
    xyxy[:, 0] = boxes_raw[:, 0] - boxes_raw[:, 2] / 2
    xyxy[:, 1] = boxes_raw[:, 1] - boxes_raw[:, 3] / 2
    xyxy[:, 2] = boxes_raw[:, 0] + boxes_raw[:, 2] / 2
    xyxy[:, 3] = boxes_raw[:, 1] + boxes_raw[:, 3] / 2

    # 去掉 pad → 原始坐标
    xyxy[:, 0] = (xyxy[:, 0] - dw) / r
    xyxy[:, 1] = (xyxy[:, 1] - dh) / r
    xyxy[:, 2] = (xyxy[:, 2] - dw) / r
    xyxy[:, 3] = (xyxy[:, 3] - dh) / r

    xyxy[:, 0] = np.clip(xyxy[:, 0], 0, fw)
    xyxy[:, 1] = np.clip(xyxy[:, 1], 0, fh)
    xyxy[:, 2] = np.clip(xyxy[:, 2], 0, fw)
    xyxy[:, 3] = np.clip(xyxy[:, 3], 0, fh)

    indices = cv2.dnn.NMSBoxes(xyxy.tolist(), scores.tolist(),
                                CONF_THRES, IOU_THRES)
    if len(indices) == 0:
        return [], [], []

    keep = indices.flatten()
    return xyxy[keep], scores[keep], class_ids[keep]


def draw_boxes(frame, boxes, scores, class_ids):
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names.get(int(cls_id), '?')} {score:.2f}"
        color = colors[int(cls_id) % len(colors)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame


def main():
    print(f"加载: {MODEL_XML}")
    core = ov.Core()
    compiled = core.compile_model(core.read_model(MODEL_XML), "AUTO")
    infer = compiled.create_infer_request()

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"无法打开摄像头 {CAM_INDEX}")
        return

    print("按 'q' 退出...")
    fps_tick = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取帧失败")
            break

        blob, pad = preprocess(frame)
        output = list(infer.infer([blob]).values())[0]
        boxes, scores, class_ids = postprocess(output, pad, frame.shape)

        annotated = draw_boxes(frame, boxes, scores, class_ids)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - fps_tick)
        fps_tick = cv2.getTickCount()
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("OpenVINO Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
