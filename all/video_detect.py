"""
批量检测视频 — 三窗口 + 导出
  原视频 | YOLO检测 | OpenVINO检测
按 's' 保存当前视频的检测结果，按 'n' 下一个，按 'q' 退出
"""
import cv2
import numpy as np
import openvino as ov
from ultralytics import YOLO
from pathlib import Path

# ============================================================
MODEL_PT = r"runs\detect\opt_drone\weights\best.pt"
MODEL_OV = r"model_openvino\best.xml"
VIDEO_DIR = Path("test_videos")
OUTPUT_DIR = Path("test_videos")
CONF_THRES = 0.5
IMGSZ = 640
# ============================================================

# --- PyTorch 模型 ---
model_pt = YOLO(MODEL_PT)

# --- OpenVINO 模型 ---
core = ov.Core()
compiled = core.compile_model(core.read_model(MODEL_OV), "CPU")
infer_ov = compiled.create_infer_request()


def letterbox(img, s=640, c=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(s / h, s / w)
    nh, nw = int(h * r), int(w * r)
    dw, dh = (s - nw) // 2, (s - nh) // 2
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(img, dh, s - nh - dh, dw, s - nw - dw,
                              cv2.BORDER_CONSTANT, value=c), (r, dw, dh)


def preprocess(frame):
    img, pad = letterbox(frame, IMGSZ)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0, pad


def detect_ov(frame):
    blob, pad = preprocess(frame)
    output = list(infer_ov.infer([blob]).values())[0]  # (1, 11, 8400)
    output = output[0].T
    bx = output[:, :4]
    sr = output[:, 4:]; ci = sr.argmax(1); sc = sr.max(1)
    m = sc > CONF_THRES; bx, sc, ci = bx[m], sc[m], ci[m]
    if len(bx) == 0: return [], [], []
    r, dw, dh = pad; fh, fw = frame.shape[:2]
    xy = np.zeros_like(bx)
    xy[:,0]=bx[:,0]-bx[:,2]/2; xy[:,1]=bx[:,1]-bx[:,3]/2
    xy[:,2]=bx[:,0]+bx[:,2]/2; xy[:,3]=bx[:,1]+bx[:,3]/2
    xy[:,0]=np.clip((xy[:,0]-dw)/r,0,fw); xy[:,1]=np.clip((xy[:,1]-dh)/r,0,fh)
    xy[:,2]=np.clip((xy[:,2]-dw)/r,0,fw); xy[:,3]=np.clip((xy[:,3]-dh)/r,0,fh)
    idx = cv2.dnn.NMSBoxes(xy.tolist(), sc.tolist(), CONF_THRES, 0.45)
    if len(idx) == 0: return [], [], []
    k = idx.flatten(); return xy[k], sc[k], ci[k]


names = {0: "H", 1: "tent", 2: "car", 3: "bridge", 4: "pillbox", 5: "tank", 6: "Red_cross"}
colors = [(0,255,255),(0,165,255),(0,255,0),(255,0,0),(255,0,255),(0,200,200),(255,255,255)]

videos = sorted(VIDEO_DIR.glob("*.mp4")) + sorted(VIDEO_DIR.glob("*.avi"))
print(f"找到 {len(videos)} 个视频: {[v.name for v in videos]}")

for vi, vp in enumerate(videos):
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        print(f"无法打开: {vp.name}")
        continue

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out_path = OUTPUT_DIR / f"{vp.stem}_detected.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print(f"\n[{vi+1}/{len(videos)}] {vp.name} → {out_path.name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("  结束")
            break

        # PyTorch
        r_pt = model_pt.predict(frame, imgsz=IMGSZ, conf=CONF_THRES, verbose=False)
        ann_pt = r_pt[0].plot()

        # OpenVINO
        ann_ov = frame.copy()
        bx, sc, ci = detect_ov(frame)
        for box, s, c in zip(bx, sc, ci):
            x1, y1, x2, y2 = map(int, box)
            lbl = f"{names.get(int(c), '?')} {s:.2f}"
            co = colors[int(c) % 7]
            cv2.rectangle(ann_ov, (x1, y1), (x2, y2), co, 2)
            cv2.putText(ann_ov, lbl, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, co, 2)

        # 保存
        writer.write(ann_ov)

        # 三窗口
        cv2.imshow("Original", frame)
        cv2.imshow("PyTorch", ann_pt)
        cv2.imshow("OpenVINO", ann_ov)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release(); writer.release()
            cv2.destroyAllWindows()
            print("退出")
            exit()
        elif key == ord('n'):
            print("  跳过")
            break

    cap.release()
    writer.release()
    print(f"  保存: {out_path}")

cv2.destroyAllWindows()
print("\n全部完成!")
