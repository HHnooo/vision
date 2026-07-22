import cv2
from ultralytics import YOLO

MODEL_PATH = r'C:\Users\Admin\Desktop\yolo_result\runs\detect\opt_drone\weights\best.pt'
CAM_INDEX = 0
CONF_THRES = 0.5

model = YOLO(MODEL_PATH, task='detect')

# 只使用 DirectShow 后端（你拍照时用的就是这个）
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"无法打开摄像头 {CAM_INDEX}，请尝试其他索引")
    exit()

print("按 'q' 退出...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("读取帧失败")
        break
    results = model.predict(frame, imgsz=640, conf=CONF_THRES, verbose=False)
    annotated = results[0].plot()
    cv2.imshow("YOLOv8 实时检测", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()