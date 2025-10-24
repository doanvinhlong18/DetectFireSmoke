import os
import sys
import argparse
import math
import random
from glob import glob
from typing import List, Tuple

from PIL import Image
import numpy as np
import cv2

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# ! SỬA: Import class TinyYOLO từ file của bạn.
# ! (Giả sử file đó tên là tiny_yolo_model.py, bạn hãy đổi tên cho đúng)
from tiny_yolo_from_scratch import TinyYOLO

# ===== 1. Cấu hình (PHẢI GIỐNG HỆT FILE TRAIN) =====
# ! SỬA: Kích thước ảnh là 640
IMAGE_SIZE = 416
# ! SỬA: Số lượng class là 3
NUM_CLASSES = 3
CLASS_NAMES = ["smoke", "fire", "other"]
# ! SỬA: Các hằng số phải khớp file train
ANCHORS = [(22,29), (72,85), (348,328)]
NUM_ANCHORS = len(ANCHORS)
STRIDE = 32
GRID_SIZE = IMAGE_SIZE // STRIDE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")

# ===== 2. Load model =====
# ! SỬA: Phải truyền num_classes và anchors vào constructor
model = TinyYOLO(num_classes=NUM_CLASSES, anchors=ANCHORS).to(DEVICE)
model.load_state_dict(torch.load("runs/exp/best.pt", map_location=DEVICE))
model.eval()
print("✅ Model loaded successfully!")


# ===== 3. Hàm decode, NMS, draw (COPY TỪ FILE TRAIN) =====
# ! THÊM: Đây là hàm decode_pred CHUẨN từ file train của bạn
def decode_pred(pred_tensor, img_size=IMAGE_SIZE, stride=STRIDE, anchors=ANCHORS, conf_thres=0.3):
    # pred_tensor: B, out_ch, G, G  (torch tensor on CPU or GPU)
    B, C, G, _ = pred_tensor.shape
    A = len(anchors)
    pred = pred_tensor.view(B, A, 5 + NUM_CLASSES, G, G).permute(0, 1, 3, 4, 2).contiguous()  # B,A,G,G,5+nc

    device = pred.device
    grid_y, grid_x = torch.meshgrid(torch.arange(G, device=device), torch.arange(G, device=device), indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=-1).float()  # G,G,2
    anchors_tensor = torch.tensor(anchors, device=device).float() / stride

    output_batches = []
    for b in range(B):
        preds = pred[b]  # A,G,G,5+nc
        boxes = []
        for a in range(A):
            p = preds[a]  # G,G,5+nc
            tx = p[..., 0];
            ty = p[..., 1];
            tw = p[..., 2];
            th = p[..., 3]
            conf = torch.sigmoid(p[..., 4])
            cls_logits = p[..., 5:]
            cls_prob = torch.sigmoid(cls_logits)  # (G,G,NUM_CLASSES)

            bx = (torch.sigmoid(tx) + grid[..., 0]) / G
            by = (torch.sigmoid(ty) + grid[..., 1]) / G
            bw = torch.exp(tw) * anchors_tensor[a, 0] / G
            bh = torch.exp(th) * anchors_tensor[a, 1] / G

            bx = bx.view(-1).cpu().numpy()
            by = by.view(-1).cpu().numpy()
            bw = bw.view(-1).cpu().numpy()
            bh = bh.view(-1).cpu().numpy()
            conf_np = conf.view(-1).cpu().numpy()
            cls_prob_np = cls_prob.view(-1, NUM_CLASSES).cpu().numpy()

            for i in range(len(bx)):
                best_cls = int(np.argmax(cls_prob_np[i]))
                score = float(conf_np[i] * cls_prob_np[i, best_cls])
                if score < conf_thres:
                    continue
                cx = bx[i];
                cy = by[i];
                w = bw[i];
                h = bh[i]
                # to absolute pixel coords
                x1 = int(max(0, (cx - w / 2) * img_size))
                y1 = int(max(0, (cy - h / 2) * img_size))
                x2 = int(min(img_size, (cx + w / 2) * img_size))
                y2 = int(min(img_size, (cy + h / 2) * img_size))
                boxes.append([x1, y1, x2, y2, score, best_cls])
        if len(boxes) == 0:
            output_batches.append(np.zeros((0, 6)))
        else:
            output_batches.append(np.array(boxes))
    return output_batches


# ! THÊM: Hàm NMS từ file train
def nms_numpy(boxes: np.ndarray, iou_thresh=0.45):
    # boxes: N x [x1,y1,x2,y2,score,cls]
    if boxes.size == 0:
        return boxes
    keep = []
    idxs = np.argsort(-boxes[:, 4])
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        ious = []
        for j in rest:
            xx1 = max(boxes[i, 0], boxes[j, 0])
            yy1 = max(boxes[i, 1], boxes[j, 1])
            xx2 = min(boxes[i, 2], boxes[j, 2])
            yy2 = min(boxes[i, 3], boxes[j, 3])
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            inter = w * h
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_j = (boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1])
            union = area_i + area_j - inter + 1e-9
            iou = inter / union
            ious.append(iou)
        ious = np.array(ious)
        idxs = idxs[1:][ious < iou_thresh]
    return boxes[keep]


# ! THÊM: Hàm vẽ box từ file train
def draw_detections(img, boxes):
    # boxes: N x [x1,y1,x2,y2,score,cls]
    color_map = {0: (0, 165, 255), 1: (0, 0, 255), 2: (0, 255, 0)}  # fire=red, smoke=orange, other=green
    for b in boxes:
        x1, y1, x2, y2, score, cls_id = b
        cls_id = int(cls_id)
        color = color_map.get(cls_id, (255, 255, 255))
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{CLASS_NAMES[cls_id]} {score:.2f}"
        cv2.putText(img, label, (int(x1), max(15, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


# ===== 4. Hàm predict 1 ảnh (SỬA LẠI HOÀN TOÀN) =====
def predict_image(img_path, model, device, conf=0.5, iou_thres=0.45):
    img_cv = cv2.imread(img_path)
    if img_cv is None:
        print("❌ Không thể đọc ảnh:", img_path)
        return

    # Lưu kích thước gốc để vẽ lại cho đúng
    h0, w0 = img_cv.shape[:2]

    # --- CHUẨN HÓA ẢNH (GIỐNG HỆT FILE TRAIN) ---
    # 1. Resize về IMAGE_SIZE
    img_resized = cv2.resize(img_cv, (IMAGE_SIZE, IMAGE_SIZE))
    # 2. Chuyển từ BGR (cv2) sang RGB (PIL)
    img_pil = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    # 3. Chuyển sang Tensor (scale 0-1)
    img_tensor = TF.to_tensor(img_pil)
    # 4. Normalize (giống hệt file train)
    img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # 5. Thêm batch dimension và đưa lên device
    img_tensor = img_tensor.unsqueeze(0).to(device)
    # ---------------------------------------------

    # Dự đoán
    with torch.no_grad():
        output = model(img_tensor)  # output shape: [1, 24, 20, 20]

    # ===== THÊM ĐOẠN DEBUG NÀY VÀO ĐÂY =====
    # Reshape output để dễ dàng truy cập vào confidence score
    pred_reshaped = output.view(1, NUM_ANCHORS, 5 + NUM_CLASSES, GRID_SIZE, GRID_SIZE)
    # Lấy ra confidence score của tất cả các box (nó ở vị trí thứ 5, index 4)
    confidences_raw = pred_reshaped[:, :, 4, :, :]
    # Áp dụng hàm sigmoid để chuyển từ logit sang xác suất (0-1)
    confidences_prob = torch.sigmoid(confidences_raw)
    # Tìm giá trị confidence cao nhất
    max_conf = confidences_prob.max().item()
    print(f"🕵️‍♂️ DEBUG: Max confidence score from raw output = {max_conf:.4f}")
    # =======================================

    # GIẢI MÃ (Dùng hàm của file train)
    boxes_batch = decode_pred(output, img_size=IMAGE_SIZE, stride=STRIDE, anchors=ANCHORS, conf_thres=conf)
    boxes = boxes_batch[0]  # Lấy kết quả của ảnh đầu tiên (và duy nhất)

    # LỌC QUA NMS
    boxes = nms_numpy(boxes, iou_thresh=iou_thres)
    print(f"🔍 Found {len(boxes)} boxes")

    # RESCALE tọa độ về kích thước ảnh GỐC (h0, w0)
    if boxes.size != 0:
        scale_x = w0 / IMAGE_SIZE
        scale_y = h0 / IMAGE_SIZE
        boxes[:, 0] = boxes[:, 0] * scale_x
        boxes[:, 1] = boxes[:, 1] * scale_y
        boxes[:, 2] = boxes[:, 2] * scale_x
        boxes[:, 3] = boxes[:, 3] * scale_y

        # Cắt/Giới hạn tọa độ trong kích thước ảnh gốc
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)

    # Vẽ box lên ảnh GỐC (img_cv), không phải ảnh resized
    img_boxed = draw_detections(img_cv.copy(), boxes)

    cv2.imshow("Prediction", img_boxed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ===== 5. Chạy thử =====
if __name__ == "__main__":
    # ! SỬA: Đảm bảo đường dẫn ảnh của bạn là chính xác
    img_path = r"datasets/fire-smoke/test/images/fire2_mp4-70_jpg.rf.fbc2015ea4e7f063e0c19c4c4a09818c.jpg"

    # Kiểm tra xem file có tồn tại không
    if not os.path.exists(img_path):
        print(f"❌ LỖI: Không tìm thấy file ảnh tại: {img_path}")
    else:
        # ! SỬA: Truyền model và device vào hàm
        predict_image(img_path, model, DEVICE)