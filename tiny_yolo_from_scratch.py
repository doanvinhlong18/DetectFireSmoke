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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as T

# -------------------------
# Config / Hyperparams
# -------------------------
IMAGE_SIZE = 416         # input size (square)
BATCH_SIZE = 4           # reduce if OOM
EPOCHS = 60
LR = 1e-3
NUM_CLASSES = 3          # 0,1,2 per bạn
CLASS_NAMES = ["smoke", "fire", "other"]
DATA_DIR = "./datasets/fire-smoke"  # expects train/valid/test subfolders with images/ & labels/
PRINT_EVERY = 10

ANCHORS = [(22,29), (72,85), (348,328)]  # in pixels (relative to IMAGE_SIZE before stride normalization)
NUM_ANCHORS = len(ANCHORS)
STRIDE = 32
GRID_SIZE = IMAGE_SIZE // STRIDE  # e.g., 640/32 = 20

os.makedirs("runs/exp", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------------------
# Utils
# -------------------------
def iou_wh(box_wh, anchor_wh):
    # IoU for widths/heights centered at same point
    w1, h1 = box_wh
    w2, h2 = anchor_wh
    inter_w = min(w1, w2)
    inter_h = min(h1, h2)
    inter = max(inter_w, 0) * max(inter_h, 0)
    area1 = max(w1,0) * max(h1,0)
    area2 = max(w2,0) * max(h2,0)
    union = area1 + area2 - inter + 1e-9
    return inter / union

def xywh_to_xyxy_normalized(cx, cy, w, h):
    # inputs normalized 0..1 -> returns x1,y1,x2,y2 normalized
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    return x1, y1, x2, y2

def clamp01(x):
    return max(0.0, min(1.0, x))

# -------------------------
# Dataset
# -------------------------
class YoloDataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=IMAGE_SIZE, augment=False):
        self.images = sorted(glob(os.path.join(images_dir, "*.jpg")) + glob(os.path.join(images_dir, "*.png")))
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.augment = augment
        # augmentation helpers
        if self.augment:
            self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        base = os.path.basename(img_path)
        name = os.path.splitext(base)[0]
        label_path = os.path.join(self.labels_dir, name + ".txt")
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls = int(parts[0])
                    cx = float(parts[1])
                    cy = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    boxes.append([cls, cx, cy, w, h])

        # augmentation (simple)
        if self.augment:
            # random scale between 0.8 and 1.2, then resize back
            if random.random() < 0.9:
                scale = random.uniform(0.8, 1.2)
                new_w = int(self.img_size * scale)
                new_h = int(self.img_size * scale)
                img = img.resize((new_w, new_h))
                img = img.resize((self.img_size, self.img_size))

            if random.random() < 0.8:
                img = self.color_jitter(img)

            if random.random() < 0.05:
                img = TF.to_grayscale(img, num_output_channels=3)

            if random.random() < 0.5:
                img = TF.hflip(img)
                # update box cx
                for i in range(len(boxes)):
                    boxes[i][1] = 1.0 - boxes[i][1]

        # final resize + to tensor + normalize
        img = img.resize((self.img_size, self.img_size))
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

        if len(boxes) == 0:
            labels = torch.zeros((0,5), dtype=torch.float32)
        else:
            labels = torch.tensor(boxes, dtype=torch.float32)
        return img, labels, img_path

def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    paths = [b[2] for b in batch]
    return imgs, labels, paths

# -------------------------
# Model (TinyYOLO-like)
# -------------------------
class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class TinyYOLO(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, anchors=ANCHORS):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = torch.tensor(anchors, dtype=torch.float32)  # pixel anchors
        self.layer1 = nn.Sequential(
            ConvBNAct(3,16,3,1,1), nn.MaxPool2d(2,2),
            ConvBNAct(16,32,3,1,1), nn.MaxPool2d(2,2),
            ConvBNAct(32,64,3,1,1), nn.MaxPool2d(2,2),
            ConvBNAct(64,128,3,1,1), nn.MaxPool2d(2,2),
            ConvBNAct(128,256,3,1,1), nn.MaxPool2d(2,2),
        )
        out_ch = NUM_ANCHORS * (5 + num_classes)
        self.head = nn.Conv2d(256, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        feat = self.layer1(x)
        out = self.head(feat)
        return out

# -------------------------
# Loss components
# -------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * ((1 - pt) ** self.gamma) * bce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def ciou_loss_tensor(pred_boxes, target_boxes):
    # pred_boxes, target_boxes: (N,4) in absolute coords x1,y1,x2,y2
    eps = 1e-7
    x1 = torch.max(pred_boxes[:,0], target_boxes[:,0])
    y1 = torch.max(pred_boxes[:,1], target_boxes[:,1])
    x2 = torch.min(pred_boxes[:,2], target_boxes[:,2])
    y2 = torch.min(pred_boxes[:,3], target_boxes[:,3])
    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h
    area_p = (pred_boxes[:,2]-pred_boxes[:,0]).clamp(min=0) * (pred_boxes[:,3]-pred_boxes[:,1]).clamp(min=0)
    area_t = (target_boxes[:,2]-target_boxes[:,0]).clamp(min=0) * (target_boxes[:,3]-target_boxes[:,1]).clamp(min=0)
    union = area_p + area_t - inter + eps
    iou = inter / union

    # center dist
    cx_p = (pred_boxes[:,0] + pred_boxes[:,2]) / 2
    cy_p = (pred_boxes[:,1] + pred_boxes[:,3]) / 2
    cx_t = (target_boxes[:,0] + target_boxes[:,2]) / 2
    cy_t = (target_boxes[:,1] + target_boxes[:,3]) / 2
    center_dist_sq = (cx_p - cx_t)**2 + (cy_p - cy_t)**2

    # enclosure
    en_x1 = torch.min(pred_boxes[:,0], target_boxes[:,0])
    en_y1 = torch.min(pred_boxes[:,1], target_boxes[:,1])
    en_x2 = torch.max(pred_boxes[:,2], target_boxes[:,2])
    en_y2 = torch.max(pred_boxes[:,3], target_boxes[:,3])
    c_diag_sq = (en_x2 - en_x1)**2 + (en_y2 - en_y1)**2 + eps

    # v (aspect)
    w_p = (pred_boxes[:,2]-pred_boxes[:,0]).clamp(min=eps)
    h_p = (pred_boxes[:,3]-pred_boxes[:,1]).clamp(min=eps)
    w_t = (target_boxes[:,2]-target_boxes[:,0]).clamp(min=eps)
    h_t = (target_boxes[:,3]-target_boxes[:,1]).clamp(min=eps)
    v = (4 / (math.pi**2)) * (torch.atan(w_t / h_t) - torch.atan(w_p / h_p))**2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    ciou = iou - (center_dist_sq / c_diag_sq) - alpha * v
    return 1 - ciou  # loss

class YoloLoss(nn.Module):
    def __init__(self, anchors=ANCHORS, img_size=IMAGE_SIZE, stride=STRIDE, device=DEVICE):
        super().__init__()
        self.anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
        self.img_size = img_size
        self.stride = stride
        pos_weight_tensor = torch.tensor([4.0], device=device)
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight_tensor)
        self.focal_cls = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')

    def forward(self, preds, targets):
        # preds: B, out_ch, G, G
        B, C, G, _ = preds.shape
        A = NUM_ANCHORS
        device = preds.device
        pred = preds.view(B, A, 5 + NUM_CLASSES, G, G).permute(0,1,3,4,2).contiguous()  # B,A,G,G,5+nc

        # build target tensor same shape
        target_tensor = torch.zeros_like(pred, device=device)

        for b in range(B):
            t = targets[b]
            if t.numel() == 0:
                continue
            for box in t:
                cls, cx, cy, w, h = box.tolist()
                gx = cx * G; gy = cy * G
                gi = int(gx); gj = int(gy)
                # match anchor by wh iou
                best_a = 0; best_iou = -1
                for a_idx, anc in enumerate(self.anchors):
                    anc_w = anc[0].item() / self.img_size
                    anc_h = anc[1].item() / self.img_size
                    iou = iou_wh((w,h),(anc_w,anc_h))
                    if iou > best_iou:
                        best_iou = iou; best_a = a_idx
                target_tensor[b, best_a, gj, gi, 0] = gx - gi
                target_tensor[b, best_a, gj, gi, 1] = gy - gj
                anchor_w = self.anchors[best_a,0].item() / self.img_size
                anchor_h = self.anchors[best_a,1].item() / self.img_size
                target_tensor[b, best_a, gj, gi, 2] = math.log((w + 1e-9) / (anchor_w + 1e-9))
                target_tensor[b, best_a, gj, gi, 3] = math.log((h + 1e-9) / (anchor_h + 1e-9))
                target_tensor[b, best_a, gj, gi, 4] = 1.0
                target_tensor[b, best_a, gj, gi, 5 + int(cls)] = 1.0

        # preds components
        p_tx = pred[...,0]; p_ty = pred[...,1]; p_tw = pred[...,2]; p_th = pred[...,3]
        p_obj = pred[...,4]; p_cls = pred[...,5:]
        t_tx = target_tensor[...,0]; t_ty = target_tensor[...,1]
        t_tw = target_tensor[...,2]; t_th = target_tensor[...,3]
        t_obj = target_tensor[...,4]; t_cls = target_tensor[...,5:]

        # objectness
        obj_loss = self.bce_obj(p_obj, t_obj)

        # class + bbox on positive positions
        if (t_obj > 0).sum() > 0:
            mask = t_obj > 0  # boolean mask
            cls_loss = self.focal_cls(p_cls[mask], t_cls[mask])

            # decode pred boxes and target boxes to absolute coords for CIoU
            grid_y, grid_x = torch.meshgrid(torch.arange(G, device=device), torch.arange(G, device=device), indexing='ij')
            grid = torch.stack((grid_x, grid_y), dim=-1).float()
            anchors_tensor = self.anchors.to(device) / self.stride  # normalized to grid units

            bx = (torch.sigmoid(p_tx) + grid[...,0]) / G
            by = (torch.sigmoid(p_ty) + grid[...,1]) / G
            bw = torch.exp(p_tw) * anchors_tensor[...,0].view(1, A, 1, 1) / G
            bh = torch.exp(p_th) * anchors_tensor[...,1].view(1, A, 1, 1) / G

            target_bx = (t_tx + grid[...,0]) / G
            target_by = (t_ty + grid[...,1]) / G
            target_bw = torch.exp(t_tw) * anchors_tensor[...,0].view(1, A, 1, 1) / G
            target_bh = torch.exp(t_th) * anchors_tensor[...,1].view(1, A, 1, 1) / G

            # form absolute boxes (B,A,G,G,4)
            pred_boxes = torch.stack([
                (bx - bw/2) * self.img_size,
                (by - bh/2) * self.img_size,
                (bx + bw/2) * self.img_size,
                (by + bh/2) * self.img_size
            ], dim=-1)

            tgt_boxes = torch.stack([
                (target_bx - target_bw/2) * self.img_size,
                (target_by - target_bh/2) * self.img_size,
                (target_bx + target_bw/2) * self.img_size,
                (target_by + target_bh/2) * self.img_size
            ], dim=-1)

            pred_sel = pred_boxes[mask]
            tgt_sel = tgt_boxes[mask]
            if pred_sel.numel() > 0:
                bbox_loss = ciou_loss_tensor(pred_sel.view(-1,4), tgt_sel.view(-1,4)).mean()
            else:
                bbox_loss = torch.tensor(0.0, device=device)
        else:
            cls_loss = torch.tensor(0.0, device=device)
            bbox_loss = torch.tensor(0.0, device=device)

        total = bbox_loss + cls_loss + obj_loss
        return total, {'total': total.item(), 'bbox': bbox_loss.item(), 'cls': cls_loss.item(), 'obj': obj_loss.item()}

# -------------------------
# Decode + NMS for inference
# -------------------------
def decode_pred(pred_tensor, img_size=IMAGE_SIZE, stride=STRIDE, anchors=ANCHORS, conf_thres=0.3):
    # pred_tensor: B, out_ch, G, G  (torch tensor on CPU or GPU)
    B, C, G, _ = pred_tensor.shape
    A = len(anchors)
    pred = pred_tensor.view(B, A, 5 + NUM_CLASSES, G, G).permute(0,1,3,4,2).contiguous()  # B,A,G,G,5+nc

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
            tx = p[...,0]; ty = p[...,1]; tw = p[...,2]; th = p[...,3]
            conf = torch.sigmoid(p[...,4])
            cls_logits = p[...,5:]
            cls_prob = torch.sigmoid(cls_logits)  # (G,G,NUM_CLASSES)

            bx = (torch.sigmoid(tx) + grid[...,0]) / G
            by = (torch.sigmoid(ty) + grid[...,1]) / G
            bw = torch.exp(tw) * anchors_tensor[a,0] / G
            bh = torch.exp(th) * anchors_tensor[a,1] / G

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
                cx = bx[i]; cy = by[i]; w = bw[i]; h = bh[i]
                # to absolute pixel coords
                x1 = int(max(0, (cx - w/2) * img_size))
                y1 = int(max(0, (cy - h/2) * img_size))
                x2 = int(min(img_size, (cx + w/2) * img_size))
                y2 = int(min(img_size, (cy + h/2) * img_size))
                boxes.append([x1, y1, x2, y2, score, best_cls])
        if len(boxes) == 0:
            output_batches.append(np.zeros((0,6)))
        else:
            output_batches.append(np.array(boxes))
    return output_batches

def nms_numpy(boxes: np.ndarray, iou_thresh=0.45):
    # boxes: N x [x1,y1,x2,y2,score,cls]
    if boxes.size == 0:
        return boxes
    keep = []
    idxs = np.argsort(-boxes[:,4])
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        ious = []
        for j in rest:
            xx1 = max(boxes[i,0], boxes[j,0])
            yy1 = max(boxes[i,1], boxes[j,1])
            xx2 = min(boxes[i,2], boxes[j,2])
            yy2 = min(boxes[i,3], boxes[j,3])
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            inter = w * h
            area_i = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])
            area_j = (boxes[j,2]-boxes[j,0])*(boxes[j,3]-boxes[j,1])
            union = area_i + area_j - inter + 1e-9
            iou = inter / union
            ious.append(iou)
        ious = np.array(ious)
        idxs = idxs[1:][ious < iou_thresh]
    return boxes[keep]

# -------------------------
# Training / Validation / Eval
# -------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss = 0.0
    for i, (imgs, labels, _) in enumerate(loader):
        imgs = imgs.to(device)
        labels = [l.to(device) if isinstance(l, torch.Tensor) and l.numel() > 0 else l for l in labels]
        optimizer.zero_grad()
        preds = model(imgs)
        loss, loss_dict = criterion(preds, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                pass
        total_loss += loss.item()
        if i % PRINT_EVERY == 0:
            print(f" batch {i}/{len(loader)} loss={loss.item():.4f} info={loss_dict}")
    return total_loss / max(1, len(loader))

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            labels = [l.to(device) if isinstance(l, torch.Tensor) and l.numel() > 0 else l for l in labels]
            preds = model(imgs)
            loss, _ = criterion(preds, labels)
            total_loss += loss.item()
    return total_loss / max(1, len(loader))

def evaluate(model, loader, device, conf_thresh=0.5, iou_thresh=0.5):
    model.eval()
    all_preds = []; all_gts = []
    with torch.no_grad():
        for imgs, labels, paths in loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            boxes_batch = decode_pred(preds.cpu(), img_size=IMAGE_SIZE, conf_thres=conf_thresh)
            for i, boxes in enumerate(boxes_batch):
                boxes = boxes if boxes.size == 0 else boxes[boxes[:,4] > conf_thresh]
                boxes = nms_numpy(boxes, iou_thresh=iou_thresh) if boxes.size != 0 else boxes
                gts = labels[i].cpu().numpy() if labels[i].numel() > 0 else np.zeros((0,5))
                all_preds.append(boxes)
                all_gts.append(gts)
    # compute simple metrics (precision/recall)
    tp=fp=fn=0
    for preds, gts in zip(all_preds, all_gts):
        matched = set()
        for p in preds:
            found=False
            for gi, gt in enumerate(gts):
                cls_g, cx, cy, w, h = gt
                x1g = (cx - w/2) * IMAGE_SIZE; y1g = (cy - h/2) * IMAGE_SIZE
                x2g = (cx + w/2) * IMAGE_SIZE; y2g = (cy + h/2) * IMAGE_SIZE
                # IoU
                xx1 = max(p[0], x1g); yy1 = max(p[1], y1g)
                xx2 = min(p[2], x2g); yy2 = min(p[3], y2g)
                inter = max(0, xx2-xx1)*max(0, yy2-yy1)
                area_p = (p[2]-p[0])*(p[3]-p[1])
                area_g = (x2g-x1g)*(y2g-y1g)
                iou = inter / (area_p + area_g - inter + 1e-9)
                if iou > iou_thresh and int(p[5]) == int(cls_g) and gi not in matched:
                    matched.add(gi); found=True; break
            if found:
                tp += 1
            else:
                fp += 1
        fn += max(0, len(gts) - len(matched))
    precision = tp/(tp+fp+1e-9)
    recall = tp/(tp+fn+1e-9)
    f1 = 2*precision*recall/(precision+recall+1e-9)
    print(f"Eval: Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}")
    return precision, recall, f1

# -------------------------
# Inference helper (file or webcam)
# -------------------------
def draw_detections(img, boxes):
    # boxes: N x [x1,y1,x2,y2,score,cls]
    color_map = {0:(0,165,255), 1:(0,0,255), 2:(0,255,0)}
    for b in boxes:
        x1,y1,x2,y2,score,cls_id = b
        cls_id = int(cls_id)
        color = color_map.get(cls_id, (255,255,255))
        cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
        label = f"{CLASS_NAMES[cls_id]} {score:.2f}"
        cv2.putText(img, label, (int(x1), max(15,int(y1)-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

def infer_image(model, device, source, conf=0.3, iou_thres=0.45, save=False, save_dir="runs/exp/predict"):
    model.eval()
    img = cv2.imread(source)
    if img is None:
        print("Cannot read image:", source); return
    h0, w0 = img.shape[:2]
    img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img_tensor = TF.to_tensor(Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)))
    img_tensor = TF.normalize(img_tensor, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    img_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(img_tensor)
    boxes_batch = decode_pred(pred.cpu(), img_size=IMAGE_SIZE, conf_thres=conf)
    boxes = boxes_batch[0]
    boxes = boxes if boxes.size==0 else boxes[boxes[:,4] > conf]
    boxes = nms_numpy(boxes, iou_thresh=iou_thres) if boxes.size!=0 else boxes
    # rescale coords to original image if IMAGE_SIZE != original
    if boxes.size != 0:
        scale_x = w0 / IMAGE_SIZE
        scale_y = h0 / IMAGE_SIZE
        boxes[:,0] = boxes[:,0] * scale_x
        boxes[:,1] = boxes[:,1] * scale_y
        boxes[:,2] = boxes[:,2] * scale_x
        boxes[:,3] = boxes[:,3] * scale_y
    img_out = draw_detections(img.copy(), boxes)
    cv2.imshow("Inference", img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if save:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, os.path.basename(source))
        cv2.imwrite(out_path, img_out)
        print("Saved to", out_path)

# -------------------------
# Main entrypoint
# -------------------------
def main(args):
    if args.mode == "train":
        # dataset
        train_images = os.path.join(DATA_DIR, "train", "images")
        train_labels = os.path.join(DATA_DIR, "train", "labels")
        val_images = os.path.join(DATA_DIR, "valid", "images")
        val_labels = os.path.join(DATA_DIR, "valid", "labels")
        train_ds = YoloDataset(train_images, train_labels, augment=True)
        val_ds = YoloDataset(val_images, val_labels, augment=False)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)
        model = TinyYOLO(num_classes=NUM_CLASSES, anchors=ANCHORS).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
        criterion = YoloLoss(anchors=ANCHORS, img_size=IMAGE_SIZE, stride=STRIDE, device=DEVICE)
        # Scheduler OneCycleLR (requires steps_per_epoch)
        steps_per_epoch = max(1, len(train_loader))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=steps_per_epoch, epochs=EPOCHS)
        best_val = 1e9; epochs_no_improve = 0; patience = 30
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch+1}/{EPOCHS}")
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, scheduler=scheduler)
            val_loss = validate(model, val_loader, criterion, DEVICE)
            print(f" Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
            torch.save(model.state_dict(), "runs/exp/last.pt")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), "runs/exp/best.pt")
                print(f"  Saved best.pt (val_loss: {best_val:.4f})")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"  No improvement for {epochs_no_improve} epochs")
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break

    elif args.mode == "infer":
        assert args.weights is not None, "Provide --weights path"
        model = TinyYOLO(num_classes=NUM_CLASSES, anchors=ANCHORS).to(DEVICE)
        state = torch.load(args.weights, map_location=DEVICE)
        model.load_state_dict(state)
        print("Loaded weights:", args.weights)
        infer_image(model, DEVICE, args.source, conf=args.conf, iou_thres=args.iou, save=args.save)

    elif args.mode == "test":
        assert args.weights is not None, "Provide --weights path"
        test_images = os.path.join(DATA_DIR, "test", "images")
        test_labels = os.path.join(DATA_DIR, "test", "labels")
        test_ds = YoloDataset(test_images, test_labels, augment=False)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
        model = TinyYOLO(num_classes=NUM_CLASSES, anchors=ANCHORS).to(DEVICE)
        state = torch.load(args.weights, map_location=DEVICE)
        model.load_state_dict(state)
        evaluate(model, test_loader, DEVICE, conf_thresh=args.conf, iou_thresh=args.iou)

    else:
        print("mode must be train|infer|test")

if __name__ == "__main__":
    # In thông tin debug chỉ một lần ở đây
    torch.backends.cudnn.benchmark = True
    print("----- DEBUG GPU INFO -----")
    print("Python executable:", sys.executable)
    print("Torch version:", torch.__version__, "CUDA:", torch.version.cuda)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            print("GPU name:", torch.cuda.get_device_name(0))
        except Exception as e:
            print("get_device_name error:", e)
    print("---------------------------")
    print("Using device:", DEVICE)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer", "test"], required=True)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--source", type=str, default=None, help="image path for infer")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--save", action="store_true", help="save inference result")
    args = parser.parse_args()
    main(args)
