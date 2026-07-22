"""
双窗口视频标注工具 — 原始 vs OpenVINO 检测同步对比
  空格:     播放/暂停
  → / D:   前进一帧
  ← / A:   后退一帧
  1-7:     选择类别 (H/tent/car/bridge/pillbox/tank/Red_cross)
  q:       保存当前原始帧到 undetected_frames/<类别>/
  n:       切换下一个视频
  ESC:     退出
"""
import json
import cv2
import numpy as np
import openvino as ov
from pathlib import Path

# ============================================================
VIDEO_DIR = Path("test_videos")
OUTPUT_DIR = Path("undetected_frames")
RECORD_FILE = OUTPUT_DIR / "saved_frames.json"  # 帧保存记录
MODEL_XML = r"model_openvino\best.xml"
CONF_THRES = 0.5
IOU_THRES = 0.45
IMGSZ = 640
# ============================================================

CLASS_NAMES = ["H", "tent", "car", "bridge", "pillbox", "tank", "Red_cross"]
COLORS = [
    (0, 255, 255), (0, 165, 255), (0, 255, 0),
    (255, 0, 0), (255, 0, 255), (0, 200, 200), (255, 255, 255),
]


def load_record():
    """加载保存记录"""
    if RECORD_FILE.exists():
        with open(str(RECORD_FILE), "r") as f:
            record = json.load(f)
            # 自动清理不存在的视频记录
            for vname in list(record.keys()):
                for fidx in list(record[vname].keys()):
                    record[vname][fidx] = set(record[vname][fidx])
            return record
    return {}


def save_record(record):
    """保存记录到 JSON"""
    out = {}
    for vname, frames in record.items():
        out[vname] = {}
        for fidx, classes in frames.items():
            out[vname][fidx] = sorted(list(classes))
    with open(str(RECORD_FILE), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def imwrite_cn(path, img):
    path = Path(path)
    ext = path.suffix
    success, buf = cv2.imencode(ext, img)
    if success:
        with open(str(path), "wb") as f:
            f.write(buf.tobytes())
    return success


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


def detect(infer, frame):
    img, pad = letterbox(frame, IMGSZ)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
    output = list(infer.infer([img]).values())[0][0].T
    bx = output[:, :4]
    sr = output[:, 4:]; ci = sr.argmax(1); sc = sr.max(1)
    m = sc > CONF_THRES; bx, sc, ci = bx[m], sc[m], ci[m]
    if len(bx) == 0:
        return []
    r, dw, dh = pad; fh, fw = frame.shape[:2]
    xy = np.zeros_like(bx)
    xy[:,0]=bx[:,0]-bx[:,2]/2; xy[:,1]=bx[:,1]-bx[:,3]/2
    xy[:,2]=bx[:,0]+bx[:,2]/2; xy[:,3]=bx[:,1]+bx[:,3]/2
    xy[:,0]=np.clip((xy[:,0]-dw)/r,0,fw); xy[:,1]=np.clip((xy[:,1]-dh)/r,0,fh)
    xy[:,2]=np.clip((xy[:,2]-dw)/r,0,fw); xy[:,3]=np.clip((xy[:,3]-dh)/r,0,fh)
    idx = cv2.dnn.NMSBoxes(xy.tolist(), sc.tolist(), CONF_THRES, IOU_THRES)
    if len(idx) == 0:
        return []
    return [(xy[k], sc[k], int(ci[k])) for k in idx.flatten()]


def draw_detections(frame, detections):
    out = frame.copy()
    for box, score, cls_id in detections:
        x1, y1, x2, y2 = map(int, box)
        color = COLORS[cls_id % len(COLORS)]
        label = f"{CLASS_NAMES[cls_id]} {score:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(out, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return out


def main():
    for name in CLASS_NAMES:
        (OUTPUT_DIR / name).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / f"{name}_label").mkdir(parents=True, exist_ok=True)

    videos = sorted(VIDEO_DIR.glob("*.mp4")) + sorted(VIDEO_DIR.glob("*.avi"))
    if not videos:
        print(f"[错误] {VIDEO_DIR} 中没有视频")
        return
    print(f"找到 {len(videos)} 个视频: {[v.name for v in videos]}")

    # 加载保存记录
    saved_record = load_record()
    print(f"已加载保存记录: {len(saved_record)} 个视频有保存历史\n")

    print(f"加载 OpenVINO: {MODEL_XML}")
    core = ov.Core()
    compiled = core.compile_model(core.read_model(MODEL_XML), "AUTO")
    infer = compiled.create_infer_request()
    print("模型就绪\n")

    selected_class = 0
    playing = False
    video_idx = 0
    detection_cache = {}
    saved_count = 0

    cap = cv2.VideoCapture(str(videos[video_idx]))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if fps <= 0:
        fps = 30

    print(f"加载视频: {videos[video_idx].name} ({total_frames} 帧)...")
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()
    total_frames = len(all_frames)
    print(f"共 {total_frames} 帧\n")

    frame_idx = 0
    video_name = videos[video_idx].name

    # 统计已保存数
    saved_count = 0
    for vn in saved_record.values():
        for cs in vn.values():
            saved_count += len(cs)
    print(f"历史已保存: {saved_count} 个标注\n")

    print("按键说明:")
    print("  空格 = 播放/暂停")
    print("  →/← 或 D/A = 前进/后退一帧")
    print("  1-7  = 选择类别")
    print("  q    = 保存原始帧到 undetected_frames/<类别>/ (已保存的帧会跳过)")
    print("  n    = 下一个视频")
    print("  ESC  = 退出\n")

    while True:
        frame = all_frames[frame_idx]

        # 获取或计算检测结果
        if frame_idx not in detection_cache:
            detection_cache[frame_idx] = detect(infer, frame)
        detections = detection_cache[frame_idx]

        # 原始视图
        original_view = frame.copy()
        cv2.putText(original_view, "Original", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 标注视图
        labeled_view = draw_detections(frame, detections)
        cv2.putText(labeled_view, "OpenVINO Detection", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        det_counts = {}
        for _, _, cid in detections:
            det_counts[cid] = det_counts.get(cid, 0) + 1
        det_text = " | ".join([f"{CLASS_NAMES[k]}:{v}" for k, v in sorted(det_counts.items())])
        if det_text:
            cv2.putText(labeled_view, det_text, (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        if not detections:
            cv2.putText(original_view, "UNDETECTED", (original_view.shape[1] - 160, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 该帧已保存的类别
        this_frame_saved = saved_record.get(video_name, {}).get(str(frame_idx), set())
        if this_frame_saved:
            saved_names = ", ".join(this_frame_saved)
            cv2.putText(original_view, f"Saved: {saved_names}", (10, original_view.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 并排显示
        h, w = frame.shape[:2]
        display_w = w * 2 + 10
        display_h = h + 80
        display = np.zeros((display_h, display_w, 3), dtype=np.uint8)
        display[:h, :w] = original_view
        display[:h, w + 10:w * 2 + 10] = labeled_view
        cv2.line(display, (w + 5, 0), (w + 5, h), (100, 100, 100), 2)

        # 底部信息栏
        info_y = h + 5
        bottom_h = display_h - h
        info_bg = np.zeros((bottom_h, display_w, 3), dtype=np.uint8)
        info_bg[:] = (30, 30, 30)
        display[h:display_h, :] = info_bg

        # 已保存标识
        saved_info = ""
        if this_frame_saved:
            saved_info = f" | 本帧已保存: {', '.join(sorted(this_frame_saved))}"

        info_lines = [
            f"视频: {video_name} | 帧: {frame_idx + 1}/{total_frames}{saved_info}",
            f"类别: [{selected_class + 1}] {CLASS_NAMES[selected_class]} | 状态: {'播放中' if playing else '暂停'} | 本帧检出({CLASS_NAMES[selected_class]}): {det_counts.get(selected_class, 0)} 个 | 总计: {len(detections)}",
            "1-7选类 | q保存 | 空格播放/暂停 | D/→进 A/←退 | n下个 | ESC退出",
        ]
        for i, text in enumerate(info_lines):
            color = (0, 255, 0) if i == 1 else (200, 200, 200)
            cv2.putText(display, text, (10, info_y + 20 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 进度条上标记已保存帧
        bar_x, bar_y = 10, info_y + 68
        bar_w = display_w - 20
        bar_h = 4
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        progress = int(bar_w * frame_idx / max(total_frames - 1, 1))
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + progress, bar_y + bar_h), (0, 255, 0), -1)

        # 在进度条上标记所有已保存帧位置
        if video_name in saved_record:
            for fidx_str in saved_record[video_name]:
                fidx = int(fidx_str)
                if 0 <= fidx < total_frames:
                    dot_x = bar_x + int(bar_w * fidx / max(total_frames - 1, 1))
                    cv2.circle(display, (dot_x, bar_y + bar_h // 2), 3, (0, 200, 255), -1)

        cv2.imshow("Manual Labeling — Left: Original | Right: OpenVINO", display)

        # 按键处理
        wait_ms = int(1000 / fps) if playing else 0
        key = cv2.waitKeyEx(wait_ms)

        if key == 27:  # ESC
            break

        elif key == ord(" "):  # 空格 → 播放/暂停
            playing = not playing

        elif key in (ord("d"), ord("D"), 2555904):  # D / 右键
            if frame_idx < total_frames - 1:
                frame_idx += 1

        elif key in (ord("a"), ord("A"), 2424832):  # A / 左键
            if frame_idx > 0:
                frame_idx -= 1

        elif key in [ord(str(k)) for k in range(1, 8)]:  # 1-7
            selected_class = int(chr(key)) - 1
            print(f"  选中: [{selected_class + 1}] {CLASS_NAMES[selected_class]}")

        elif key == ord('q'):  # 保存
            name = CLASS_NAMES[selected_class]
            fidx_str = str(frame_idx)

            # 初始化记录结构
            if video_name not in saved_record:
                saved_record[video_name] = {}
            if fidx_str not in saved_record[video_name]:
                saved_record[video_name][fidx_str] = set()

            # 检查是否已保存
            if name in saved_record[video_name][fidx_str]:
                print(f"  [跳过] 帧 {frame_idx + 1} 已保存过 {name}，不重复添加")
                # 红色闪烁
                flash = display.copy()
                cv2.putText(flash, f"ALREADY SAVED! {name}", (display_w // 2 - 150, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.imshow("Manual Labeling — Left: Original | Right: OpenVINO", flash)
                cv2.waitKey(300)
                continue

            # 保存图片
            folder = OUTPUT_DIR / name
            count = len(list(folder.glob("*.jpg")))
            filename = f"{name}_{count + 1:04d}.jpg"
            filepath = folder / filename
            imwrite_cn(str(filepath), frame)
            saved_record[video_name][fidx_str].add(name)
            save_record(saved_record)
            print(f"  已保存: {filename} → undetected_frames/{name}/  (帧 {frame_idx + 1})")

            # 绿色闪烁
            flash = display.copy()
            cv2.putText(flash, f"SAVED! {name}", (display_w // 2 - 120, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imshow("Manual Labeling — Left: Original | Right: OpenVINO", flash)
            cv2.waitKey(300)

        elif key == ord("n"):
            video_idx = (video_idx + 1) % len(videos)
            video_name = videos[video_idx].name
            cap = cv2.VideoCapture(str(videos[video_idx]))
            print(f"\n加载视频: {video_name}...")
            all_frames.clear()
            detection_cache.clear()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                all_frames.append(frame)
            cap.release()
            total_frames = len(all_frames)
            frame_idx = 0
            playing = False
            print(f"共 {total_frames} 帧")

        # 自动播放 → 前进帧
        if playing and key == -1:
            if frame_idx < total_frames - 1:
                frame_idx += 1
            else:
                playing = False

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
