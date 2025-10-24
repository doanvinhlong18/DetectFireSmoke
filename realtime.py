# detect_camera.py
import cv2
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import argparse
import os

# Giả sử class model và các hàm của bạn được lưu trong file `tiny_yolo_model.py`
# Hãy đổi tên file import cho đúng với dự án của bạn
from tiny_yolo_from_scratch import TinyYOLO, decode_pred, nms_numpy, draw_detections

# ===== 1. Cấu hình (PHẢI GIỐNG HỆT FILE TRAIN) =====
IMAGE_SIZE = 416
NUM_CLASSES = 3
CLASS_NAMES = ["smoke", "fire", "other"]

# 💡 QUAN TRỌNG: Hãy sử dụng bộ anchor mới bạn đã tính toán bằng K-Means ở đây
# Ví dụ: ANCHORS = [(21,35), (45,68), (92,115)]
# Tôi sẽ tạm dùng anchor cũ để code có thể chạy, nhưng bạn nên cập nhật nó
ANCHORS = [(22,29), (72,85), (348,328)]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 2. Hàm chính để xử lý Camera =====
def detect_on_camera(weights_path, conf=0.5, iou_thres=0.45):
    """
    Hàm để chạy nhận diện real-time từ webcam.
    """
    # --- Load Model ---
    print("Loading model...")
    model = TinyYOLO(num_classes=NUM_CLASSES, anchors=ANCHORS).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval() # Chuyển model sang chế độ evaluation
    print(f"✅ Model loaded successfully on {DEVICE}!")

    # --- Khởi tạo Camera ---
    # Số 0 có nghĩa là dùng webcam mặc định của máy.
    # Nếu bạn có nhiều webcam, có thể thử số 1, 2, ...
    # Hoặc thay số 0 bằng đường dẫn tới một file video, ví dụ: "my_video.mp4"
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Cannot open camera.")
        return

    print("\n🚀 Starting real-time detection... Press 'q' to quit.")

    # --- Vòng lặp Real-time ---
    while True:
        # Đọc từng khung hình từ camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Lưu kích thước gốc của frame để vẽ lại cho đúng
        h0, w0 = frame.shape[:2]

        # --- Chuẩn hóa ảnh (GIỐNG HỆT FILE TRAIN/INFERENCE) ---
        img_resized = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = TF.to_tensor(img_rgb)
        img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

        # --- Dự đoán ---
        with torch.no_grad():
            output = model(img_tensor)

        # --- Giải mã và Lọc kết quả ---
        boxes_batch = decode_pred(output, img_size=IMAGE_SIZE, conf_thres=conf)
        boxes = boxes_batch[0]
        boxes = nms_numpy(boxes, iou_thresh=iou_thres)

        # --- Rescale tọa độ về kích thước frame gốc ---
        if boxes.size > 0:
            scale_x = w0 / IMAGE_SIZE
            scale_y = h0 / IMAGE_SIZE
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] * scale_x).clip(0, w0)
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] * scale_y).clip(0, h0)

        # --- Vẽ bounding box lên frame gốc ---
        result_frame = draw_detections(frame, boxes)

        # --- Hiển thị kết quả ---
        cv2.imshow('Real-time Fire & Smoke Detection - Press Q to Exit', result_frame)

        # Đợi 1ms và kiểm tra nếu người dùng nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Dọn dẹp ---
    print("Exiting...")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time YOLO Detection from Camera")
    parser.add_argument("--weights", type=str, default="runs/exp/best.pt", help="Path to model weights file.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS.")
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"❌ ERROR: Weights file not found at: {args.weights}")
    else:
        detect_on_camera(args.weights, conf=args.conf, iou_thres=args.iou)