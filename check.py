import cv2
import os

img_path = "datasets/fire-smoke/train/images/WEB03989.jpg"  # ảnh bất kỳ
label_path = "datasets/fire-smoke/train/labels/WEB03989.txt"

img = cv2.imread(img_path)
h, w = img.shape[:2]

with open(label_path, "r") as f:
    for line in f.readlines():
        cls, x, y, bw, bh = map(float, line.split())
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, str(int(cls)), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

cv2.imshow("YOLO labels check", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# import matplotlib.pyplot as plt
# import glob
#
# label_files = glob.glob("datasets/fire-smoke/train/labels/**/*.txt", recursive=True)
# xs, ys = [], []
#
# for file in label_files:
#     with open(file) as f:
#         for line in f:
#             parts = line.split()
#             if len(parts) == 5:
#                 _, x, y, _, _ = map(float, parts)
#                 xs.append(x)
#                 ys.append(y)
#
# plt.figure(figsize=(5,5))
# plt.scatter(xs, ys, alpha=0.3, s=10, color='blue')
# plt.title("Phân bố tâm bounding box (x_center, y_center)")
# plt.xlabel("x_center")
# plt.ylabel("y_center")
# plt.grid(True)
# plt.show()
