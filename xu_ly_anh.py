import cv2
import numpy as np
import imutils
import time

# --- Tham số có thể chỉnh ---
FIRE_MIN_AREA = 500   # diện tích tối thiểu để xem là vùng lửa
SMOKE_MIN_AREA = 800  # diện tích tối thiểu để xem là vùng khói
SHOW_WINDOW = True

# HSV thresholds for fire (tunable)
# Fire thường có màu vàng/đỏ cam -> H khoảng 0-50 (0..180 in OpenCV), S cao, V cao
FIRE_LOWER = np.array([0, 120, 150])
FIRE_UPPER = np.array([50, 255, 255])

# Smoke heuristic thresholds in HSV:
# Smoke thường ít bão hòa (S thấp), có giá trị V trung bình -> S nhỏ
SMOKE_S_MAX = 80
SMOKE_V_MIN = 40
SMOKE_V_MAX = 220

# Morph kernel
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

def detect_fire(frame_hsv):
    mask = cv2.inRange(frame_hsv, FIRE_LOWER, FIRE_UPPER)
    # Clean up
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, KERNEL, iterations=2)
    return mask

def detect_smoke(frame_hsv, motion_mask):
    # Smoke: low saturation & some brightness and motion overlap
    h, s, v = cv2.split(frame_hsv)
    # candidate where S is low and V in range
    s_mask = (s <= SMOKE_S_MAX).astype('uint8') * 255
    v_mask = cv2.inRange(v, SMOKE_V_MIN, SMOKE_V_MAX)
    candidate = cv2.bitwise_and(s_mask, v_mask)
    # morphological
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, KERNEL, iterations=1)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_DILATE, KERNEL, iterations=2)
    # require motion overlap (smoke moves)
    smoke_mask = cv2.bitwise_and(candidate, motion_mask)
    return smoke_mask

def main(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Không mở được nguồn video:", source)
        return

    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
    prev_frame = None
    last_fire_time = 0
    last_smoke_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=800)
        frame_blur = cv2.GaussianBlur(frame, (5,5), 0)
        frame_hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

        # 1) Fire detection via HSV color
        fire_mask = detect_fire(frame_hsv)
        contours, _ = cv2.findContours(fire_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fire_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > FIRE_MIN_AREA:
                x,y,w,h = cv2.boundingRect(cnt)
                fire_boxes.append((x,y,w,h,area))
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
                cv2.putText(frame, f"FIRE {int(area)}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # 2) Motion mask (background sub)
        fgmask = fgbg.apply(frame_blur)
        # remove shadows (MOG2 marks shadows as 127)
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, KERNEL, iterations=1)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, KERNEL, iterations=2)

        # 3) Smoke detection (low saturation + motion)
        smoke_mask = detect_smoke(frame_hsv, fgmask)
        cnts_smoke, _ = cv2.findContours(smoke_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        smoke_boxes = []
        for cnt in cnts_smoke:
            area = cv2.contourArea(cnt)
            if area > SMOKE_MIN_AREA:
                x,y,w,h = cv2.boundingRect(cnt)
                smoke_boxes.append((x,y,w,h,area))
                cv2.rectangle(frame, (x,y), (x+w, y+h), (180,180,180), 2)
                cv2.putText(frame, f"SMOKE {int(area)}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 2)

        # 4) Optional: simple "flicker" check for fire: measure V variance in fire region
        for (x,y,w,h,area) in fire_boxes:
            roi = frame_hsv[y:y+h, x:x+w]
            v = roi[:,:,2]
            # variance of brightness: flame flicker => V variance higher
            var_v = np.var(v)
            if var_v > 500:  # threshold tunable
                cv2.putText(frame, "FLAME FLICKER", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # Alerts
        now = time.time()
        if fire_boxes:
            last_fire_time = now
        if smoke_boxes:
            last_smoke_time = now

        if now - last_fire_time < 2:
            cv2.putText(frame, "ALERT: FIRE DETECTED!", (10,30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,255), 3)
        if now - last_smoke_time < 2:
            cv2.putText(frame, "ALERT: SMOKE DETECTED!", (10,70), cv2.FONT_HERSHEY_DUPLEX, 0.9, (200,200,200), 3)

        # Show masks small for debugging
        if SHOW_WINDOW:
            small_fire = cv2.resize(fire_mask, (200,150))
            small_smoke = cv2.resize(smoke_mask, (200,150))
            small_fg = cv2.resize(fgmask, (200,150))
            # Compose a little panel
            panel = np.zeros((170, 650, 3), dtype='uint8')
            panel[10:160, 10:210] = cv2.cvtColor(small_fire, cv2.COLOR_GRAY2BGR)
            panel[10:160, 210+10:210+210] = cv2.cvtColor(small_smoke, cv2.COLOR_GRAY2BGR)
            panel[10:160, 420+10:420+210] = cv2.cvtColor(small_fg, cv2.COLOR_GRAY2BGR)
            cv2.putText(panel, "Fire mask", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
            cv2.putText(panel, "Smoke mask", (230, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            cv2.putText(panel, "Motion mask", (460, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

            # show main frame and panel
            panel = cv2.resize(panel, (frame.shape[1], panel.shape[0]))
            combined = np.vstack([frame, panel])
            cv2.imshow("Fire & Smoke Detection", combined)
        else:
            cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('p'):
            cv2.waitKey(0)  # pause on key p

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fire & Smoke detection")
    parser.add_argument('--source', type=str, default='0', help='video source: 0 for webcam or path to video file')
    args = parser.parse_args()
    src = 0 if args.source == '0' else args.source
    main(src)
