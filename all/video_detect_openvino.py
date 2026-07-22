"""
批量视频检测 — OpenVINO
读取 test_vidios/ 中的视频，使用 OpenVINO 模型检测，输出到 labeled_vidios/
"""
import cv2
import numpy as np
import openvino as ov
from pathlib import Path

# ============================================================
# 配置
# ============================================================
MODEL_XML = r"model_openvino\best.xml"
VIDEO_DIR = Path("test_videos")
OUTPUT_DIR = Path("labeled_videos")
CONF_THRES = 0.5
IOU_THRES = 0.45
IMGSZ = 640
SHOW_PREVIEW = False  # True = 显示检测窗口, False = 静默批量处理
# ============================================================

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

    # YOLOv8 output: (1, 11, 8400) -> 11 = 4(xywh) + 7(classes)
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

    # xywh -> xyxy (640x640)
    xyxy = np.zeros_like(boxes_raw)
    xyxy[:, 0] = boxes_raw[:, 0] - boxes_raw[:, 2] / 2
    xyxy[:, 1] = boxes_raw[:, 1] - boxes_raw[:, 3] / 2
    xyxy[:, 2] = boxes_raw[:, 0] + boxes_raw[:, 2] / 2
    xyxy[:, 3] = boxes_raw[:, 1] + boxes_raw[:, 3] / 2

    # 去掉 pad -> 原始坐标
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
    # 检查目录是否存在
    if not VIDEO_DIR.exists():
        print(f"[错误] 视频目录不存在: {VIDEO_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 收集所有视频
    videos = sorted(VIDEO_DIR.glob("*.mp4")) + sorted(VIDEO_DIR.glob("*.avi"))
    if not videos:
        print(f"[错误] {VIDEO_DIR} 中没有找到视频文件 (.mp4/.avi)")
        return

    print(f"找到 {len(videos)} 个视频: {[v.name for v in videos]}")

    # 加载 OpenVINO 模型
    print(f"\n加载 OpenVINO 模型: {MODEL_XML}")
    core = ov.Core()
    compiled = core.compile_model(core.read_model(MODEL_XML), "AUTO")
    infer = compiled.create_infer_request()
    print("模型加载成功\n")

    total_frames = 0
    total_objects = 0

    for vi, vp in enumerate(videos):
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            print(f"  [跳过] 无法打开: {vp.name}")
            continue

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = OUTPUT_DIR / f"{vp.stem}_labeled.mp4"
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
        )

        print(f"[{vi+1}/{len(videos)}] {vp.name} -> {out_path.name} ({total} 帧)")

        frame_count = 0
        obj_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            blob, pad = preprocess(frame)
            output = list(infer.infer([blob]).values())[0]
            boxes, scores, class_ids = postprocess(output, pad, frame.shape)

            annotated = draw_boxes(frame, boxes, scores, class_ids)
            writer.write(annotated)

            frame_count += 1
            obj_count += len(boxes)

            if SHOW_PREVIEW:
                cv2.imshow("OpenVINO", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cap.release()
                    writer.release()
                    cv2.destroyAllWindows()
                    return

        cap.release()
        writer.release()
        total_frames += frame_count
        total_objects += obj_count
        print(f"  完成: {frame_count} 帧, {obj_count} 个检测目标, 保存至 {out_path}")

    cv2.destroyAllWindows()
    print(f"\n{'='*50}")
    print(f"全部完成! 共 {len(videos)} 个视频, {total_frames} 帧, {total_objects} 个检测目标")
    print(f"输出目录: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
