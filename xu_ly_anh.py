import cv2
import numpy as np
import imutils
from collections import deque

# --- Params ---
FIRE_MIN_AREA = 500
SMOKE_MIN_AREA = 800
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
FIRE_HISTORY_LEN = 8
FIRE_HISTORY_THRESHOLD = 0.5

FIRE_LOWER = np.array([0, 50, 150])
FIRE_UPPER = np.array([60, 255, 255])

# Smoke color (gray-yellow)
SMOKE_LOWER = np.array([0, 10, 80])
SMOKE_UPPER = np.array([70, 80, 200])

# Flicker memory
flicker = {}

def update_flicker(key, v):
    if key not in flicker:
        flicker[key] = deque(maxlen=12)
    flicker[key].append(v)

def has_flicker(key):
    if key not in flicker or len(flicker[key]) < 6:
        return True
    return np.std(flicker[key]) > 8

# ================= FIRE MASK =================
def fire_mask(frame_bgr, frame_hsv):
    h,s,v = cv2.split(frame_hsv)

    # Fire basic color
    mask = cv2.inRange(frame_hsv, FIRE_LOWER, FIRE_UPPER)

    # Fire "core" = high sat + bright
    fire_core = (s > 120) & (v > 180)
    mask[~fire_core] = 0

    # Morphology (giảm close để không ăn khói)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
    mask = cv2.dilate(mask, KERNEL, iterations=1)

    return mask


# ================ SMOKE MASK =================
def smoke_mask(frame_hsv, frame_bgr, fire_mask):
    h, s, v = cv2.split(frame_hsv)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # smoke: low saturation, medium brightness
    low_sat = (s < 90)
    mid_v = (v > 80) & (v < 235)

    # edges but not too strict (khói có biên mờ, không phải 0 edge)
    edges = cv2.Canny(gray, 20, 60)
    low_edge = (edges < 40)

    smoke = (low_sat & mid_v & low_edge).astype(np.uint8) * 255

    # remove fire
    smoke[fire_mask > 0] = 0

    # smoothing shape of smoke
    smoke = cv2.morphologyEx(smoke, cv2.MORPH_OPEN, KERNEL, iterations=1)
    smoke = cv2.dilate(smoke, KERNEL, iterations=2)

    # slight blur for continuity
    smoke = cv2.GaussianBlur(smoke, (7, 7), 0)

    return smoke

# ---------------------------------------------------
# COMMON BOX EXTRACTOR
# ---------------------------------------------------
def extract_boxes(mask, frame, hsv, is_video, min_area, is_fire):
    clean = cv2.morphologyEx(mask, cv2.MORPH_ERODE, KERNEL, iterations=1)
    cnts,_ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area: continue

        (cx,cy),(w,h),_ = cv2.minAreaRect(c)
        if w < 5 or h < 5: continue

        x = int(cx - w/2)
        y = int(cy - h/2)
        w = int(w); h = int(h)

        roi = frame[y:y+h, x:x+w]
        hsv_roi = hsv[y:y+h, x:x+w]
        if roi.size == 0: continue

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()

        if is_fire:
            # flame color check
            b,g,r = cv2.split(roi)
            rgb = (r > g) & (g > b)
            if np.mean(rgb) < 0.3: continue
            if lap < 10: continue

            if is_video:
                key = (x,y,w,h)
                update_flicker(key, np.mean(hsv_roi[:,:,2]))
                if not has_flicker(key): continue

        else:
            # smoke — low texture but gradient edges exist
            if lap > 25: continue

        boxes.append((x,y,w,h))
    return boxes

# ---------------------------------------------------
def run(src):
    is_video = not (isinstance(src,str) and src.lower().endswith((".jpg",".png",".jpeg")))

    if is_video:
        cap = cv2.VideoCapture(0 if src=="0" else src)
        hist = deque(maxlen=FIRE_HISTORY_LEN)

        while True:
            ret,frame=cap.read()
            if not ret: break
            frame = imutils.resize(frame,width=900)
            blur = cv2.GaussianBlur(frame,(5,5),0)
            hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)

            fire = fire_mask(blur,hsv)
            smoke = smoke_mask(hsv, blur, fire)

            fire_boxes  = extract_boxes(fire, frame, hsv, True,  FIRE_MIN_AREA, True)
            smoke_boxes = extract_boxes(smoke, frame, hsv, True, SMOKE_MIN_AREA, False)

            hist.append(1 if len(fire_boxes)>0 else 0)
            stable = sum(hist)/len(hist) > FIRE_HISTORY_THRESHOLD

            # draw fire
            for x,y,w,h in fire_boxes:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.putText(frame,"FIRE",(x,y-5),0,0.7,(0,0,255),2)

            # draw smoke
            for x,y,w,h in smoke_boxes:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(frame,"SMOKE",(x,y-5),0,0.7,(255,0,0),2)

            if stable and len(fire_boxes)>0:
                cv2.putText(frame,"🔥 FIRE ALERT!",(10,30),0,1,(0,0,255),3)

            cv2.imshow("Fire+Smoke",frame)
            cv2.imshow("Fire Mask",fire)
            cv2.imshow("Smoke Mask",smoke)

            if cv2.waitKey(1)&0xFF==ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

    else:
        img = imutils.resize(cv2.imread(src), width=900)
        blur = cv2.GaussianBlur(img,(5,5),0)
        hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
        fire = fire_mask(blur,hsv)
        smoke = smoke_mask(hsv,blur, fire)

        fire_boxes = extract_boxes(fire,img,hsv,False,FIRE_MIN_AREA,True)
        smoke_boxes = extract_boxes(smoke,img,hsv,False,SMOKE_MIN_AREA,False)

        for x,y,w,h in fire_boxes:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(img,"FIRE",(x,y-5),0,0.7,(0,0,255),2)
        for x,y,w,h in smoke_boxes:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(img,"SMOKE",(x,y-5),0,0.7,(255,0,0),2)

        cv2.imshow("Fire+Smoke",img)
        cv2.imshow("Fire Mask",fire)
        cv2.imshow("Smoke Mask",smoke)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--source",default="0")
    run(p.parse_args().source)