"""
清空 补拍/images_add/ 中的图片和标注，
从视频每隔 10 帧提取，复制到 tent/car/bridge/pillbox/tank 文件夹（跳过 H 和 Red_cross）。
每帧复制到全部 5 个文件夹，等待手动标注。
"""
import cv2
import numpy as np
from pathlib import Path
import os


def imwrite_cn(path, img):
    """cv2.imwrite 不支持中文路径，用 imencode 绕过"""
    ext = os.path.splitext(path)[1]
    success, buf = cv2.imencode(ext, img)
    if success:
        with open(path, "wb") as f:
            f.write(buf.tobytes())
    return success

# ============================================================
BASE = Path("补拍/images_add")
VIDEO_DIRS = [Path("test_videos")]
FRAME_INTERVAL = 10  # 每隔 10 帧
# ============================================================

# 目标类别（排除 H 和 Red_cross）
TARGET_CLASSES = ["tent", "car", "bridge", "pillbox", "tank"]


def clear_dataset():
    """删除所有图片和标注文件"""
    for subdir in BASE.iterdir():
        if subdir.is_dir():
            for ext in ["*.jpg", "*.png", "*.txt"]:
                for f in subdir.glob(ext):
                    f.unlink()
                    print(f"  删除: {f.name}")
    print("清理完成\n")


def get_next_number(class_dir):
    """获取下一个文件编号"""
    existing = sorted(class_dir.glob("*.*"))
    if not existing:
        return 1
    nums = []
    for f in existing:
        try:
            nums.append(int(f.stem.split("_")[-1]))
        except ValueError:
            pass
    return max(nums) + 1 if nums else 1


def main():
    # 1. 清空数据集
    print("=" * 50)
    print("1. 清空现有数据...")
    print("=" * 50)
    clear_dataset()

    # 2. 收集所有视频
    videos = []
    for vd in VIDEO_DIRS:
        if vd.exists():
            videos += sorted(vd.glob("*.mp4")) + sorted(vd.glob("*.avi"))
    print(f"找到 {len(videos)} 个视频\n")

    total_frames = 0

    for vi, vp in enumerate(videos):
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            print(f"[跳过] 无法打开: {vp.name}")
            continue

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[{vi+1}/{len(videos)}] {vp.name} ({total} 帧)")

        frame_idx = 0
        saved = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_INTERVAL == 0:
                # 复制到全部 5 个目标文件夹
                for class_name in TARGET_CLASSES:
                    img_dir = BASE / class_name
                    img_dir.mkdir(parents=True, exist_ok=True)
                    num = get_next_number(img_dir)
                    img_path = img_dir / f"{class_name}_{num:04d}.jpg"
                    imwrite_cn(str(img_path), frame)
                saved += 1

            frame_idx += 1

        cap.release()
        total_frames += saved
        print(f"  提取 {saved} 帧 (每 {FRAME_INTERVAL} 帧), 每帧 ×{len(TARGET_CLASSES)} 文件夹")

    print("\n" + "=" * 50)
    print(f"完成! 提取 {total_frames} 帧 × {len(TARGET_CLASSES)} 文件夹 = {total_frames * len(TARGET_CLASSES)} 张图片")
    for name in TARGET_CLASSES:
        count = len(list((BASE / name).glob("*.jpg")))
        print(f"  {name:10s}: {count} 张")


if __name__ == "__main__":
    main()
