#!/usr/bin/env python3
"""
USB摄像头视频录制脚本
按 S 开始录制, 按 Q 停止录制, 按 ESC 退出
视频自动编号为 vidio0001.mp4, vidio0002.mp4, ... 保存在 ../vidios/ 目录
"""

import cv2
import os
import sys
import time

# ==================== 配置参数 ====================
DEVICE_INDEX = 0          # 摄像头设备号
FRAME_WIDTH = 640         # 分辨率宽
FRAME_HEIGHT = 480        # 分辨率高
FPS = 30.0                # 帧率
VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "all", "test_videos")
# =================================================


def get_next_filename():
    """自动编号 vidio0001.mp4, vidio0002.mp4, ..."""
    os.makedirs(VIDEO_DIR, exist_ok=True)
    index = 1
    while True:
        filepath = os.path.join(VIDEO_DIR, f"vidio{index:04d}.mp4")
        if not os.path.exists(filepath):
            return filepath
        index += 1


def main():
    # 打开摄像头
    cap = cv2.VideoCapture(DEVICE_INDEX)
    if not cap.isOpened():
        print(f"[错误] 无法打开摄像头 /dev/video{DEVICE_INDEX}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    print(f"===== 视频录制脚本 =====")
    print(f"摄像头: /dev/video{DEVICE_INDEX}")
    print(f"分辨率: {FRAME_WIDTH}x{FRAME_HEIGHT}, FPS: {FPS}")
    print(f"保存目录: {VIDEO_DIR}")
    print(f"S: 开始录制 | Q: 停止录制 | ESC: 退出")
    print(f"========================")

    writer = None
    recording = False
    frame_count = 0        # 已录制帧数

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[警告] 摄像头帧为空, 等待...")
            time.sleep(0.1)
            continue

        # 录制中: 写入视频文件
        if recording and writer is not None:
            writer.write(frame)
            frame_count += 1
            # 右上角红点 + REC + 帧计数
            cv2.circle(frame, (frame.shape[1] - 30, 25), 8, (0, 0, 255), -1)
            cv2.putText(frame, f"REC {frame_count}", (frame.shape[1] - 130, 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 左上角提示
        cv2.putText(frame, "S: Start | Q: Stop | ESC: Exit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Video Recorder", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') or key == ord('S'):
            if not recording:
                filepath = get_next_filename()
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(filepath, fourcc, FPS,
                                         (frame.shape[1], frame.shape[0]))
                if writer.isOpened():
                    recording = True
                    frame_count = 0
                    print(f"● 开始录制: {filepath}")
                else:
                    print(f"[错误] 无法创建视频文件: {filepath}")

        elif key == ord('q') or key == ord('Q'):
            if recording:
                recording = False
                frame_count = 0
                writer.release()
                writer = None
                print(f"■ 录制停止")

        elif key == 27:  # ESC
            if recording:
                writer.release()
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"已退出, 视频目录: {VIDEO_DIR}")
    for f in sorted(os.listdir(VIDEO_DIR)):
        size = os.path.getsize(os.path.join(VIDEO_DIR, f))
        print(f"  {f}  ({size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
