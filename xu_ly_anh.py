import cv2
import numpy as np
import imutils
from collections import deque

# --- Params ---
FIRE_MIN_AREA = 400
SMOKE_MIN_AREA = 900

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
KERNEL_BIG = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))

HISTORY_FRAMES = 6
FIRE_STABLE_RATIO = 0.5

flicker = {}

def update_flicker(key,val):
    if key not in flicker:
        flicker[key]=deque(maxlen=10)
    flicker[key].append(val)

def has_flicker(key):
    if key not in flicker or len(flicker[key])<5: return True
    return np.std(flicker[key])>6

# ================= FIRE MASK ==================
def fire_mask(frame_bgr, hsv):
    h,s,v = cv2.split(hsv)

    # --- Fire Core (real fire) ---
    core = (h < 25) & (s > 160) & (v > 200)

    # --- Fire Halo (reflection) ---
    halo = (h < 35) & (s > 80) & (v > 160)

    core = core.astype(np.uint8) * 255
    halo = halo.astype(np.uint8) * 255

    # blur detect
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)

    # reflection = bright but low texture
    reflection = (halo > 0) & (lap < 25)
    halo[reflection] = 0

    # Grow core slightly
    core = cv2.dilate(core, KERNEL, iterations=1)

    # final fire = core OR halo(not removed)
    fire = cv2.bitwise_or(core, halo)

    fire = cv2.morphologyEx(fire, cv2.MORPH_OPEN, KERNEL, iterations=1)
    fire = cv2.dilate(fire, KERNEL, iterations=1)

    return fire



# ================= SMOKE MASK =================
def smoke_mask(hsv, bgr, fire_m):
    h, s, v = cv2.split(hsv)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # base smoke condition
    low_sat = (s < 95)
    mid_val = (v > 60) & (v < 230)

    edges = cv2.Canny(gray, 10, 35)
    soft_edge = (edges < 25)

    smoke = (low_sat & mid_val & soft_edge).astype(np.uint8) * 255

    # --- remove fire core strongly
    fire_core = (fire_m == 255) & (s > 120)
    smoke[fire_core] = 0

    # --- remove fire halo (bright + close to fire)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(60,60))
    fire_expand = cv2.dilate(fire_m, kernel, iterations=2)

    too_bright = (v > 200)
    smoke[fire_expand == 255] = 0
    smoke[too_bright & (fire_expand == 255)] = 0

    # morphological refine
    smoke = cv2.morphologyEx(smoke, cv2.MORPH_OPEN, KERNEL, iterations=1)
    smoke = cv2.dilate(smoke, KERNEL, iterations=1)
    smoke = cv2.GaussianBlur(smoke, (5,5), 0)
    smoke = (smoke > 90).astype(np.uint8) * 255

    return smoke


# ================= BOX EXTRACTION =============
def get_boxes(mask, frame, hsv, is_video, min_area, is_fire):

    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]

    for c in cnts:
        area=cv2.contourArea(c)
        if area<min_area: continue
        x,y,w,h=cv2.boundingRect(c)

        roi = frame[y:y+h, x:x+w]
        hsv_roi=hsv[y:y+h, x:x+w]
        if roi.size==0: continue

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()

        if is_fire:
            b,g,r = cv2.split(roi)
            rgb = (r>g)&(g>b)
            if np.mean(rgb)<0.3: continue
            if lap<40: continue
            if is_video:
                key=(x,y,w,h)
                update_flicker(key, np.mean(hsv_roi[:,:,2]))
                if not has_flicker(key): continue
                if np.std(flicker[key]) < 6:
                    continue  # skip reflection
        else:
            # smoke: soft texture but not blank region
            if lap > 80: continue
            if np.mean(hsv_roi[:,:,1]) > 110: continue

        boxes.append((x,y,w,h))

    return boxes

# ================= MAIN =======================
def run(src):
    video = not (isinstance(src,str) and src.lower().endswith((".jpg",".png",".jpeg")))
    history = deque(maxlen=HISTORY_FRAMES)

    cap = None
    if video:
        cap = cv2.VideoCapture(0 if src=="0" else src)

    while True:
        if video:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = cv2.imread(src)

        frame = imutils.resize(frame,width=900)
        blur=cv2.GaussianBlur(frame,(5,5),0)
        hsv=cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)

        fireM = fire_mask(blur, hsv)
        smokeM= smoke_mask(hsv, blur, fireM)

        fire_boxes = get_boxes(fireM, frame, hsv, video, FIRE_MIN_AREA, True)
        smoke_boxes=get_boxes(smokeM,frame,hsv,video,SMOKE_MIN_AREA,False)

        fire_detect = len(fire_boxes)>0
        smoke_detect = len(smoke_boxes)>0
        history.append(1 if fire_detect else 0)
        stable = sum(history)/len(history) > FIRE_STABLE_RATIO

        for x,y,w,h in fire_boxes:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(frame,"FIRE",(x,y-5),0,0.7,(0,0,255),2)

        for x,y,w,h in smoke_boxes:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(frame,"SMOKE",(x,y-5),0,0.7,(255,0,0),2)

        if stable and fire_detect:
            cv2.putText(frame,"FIRE ALERT!",(10,30),0,1,(0,0,255),3)
        if stable and smoke_detect:
            cv2.putText(frame,"SMOKE ALERT!",(10,70),0,1,(200,0,0),3)

        cv2.imshow("Fire+Smoke",frame)
        cv2.imshow("Fire Mask",fireM)
        cv2.imshow("Smoke Mask",smokeM)

        key=cv2.waitKey(1 if video else 0)
        if key==ord('q'): break
        if not video: break

    if video: cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--source",default="0")
    run(p.parse_args().source)
