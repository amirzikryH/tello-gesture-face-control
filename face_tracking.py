import cv2
import numpy as np
import time
from djitellopy import tello
import KeyPressModule as kp
import cvzone
from ultralytics import YOLO

# -------------------- INIT --------------------
kp.init()

myDrone = tello.Tello()
myDrone.connect()
print(f"Battery: {myDrone.get_battery()}%")

myDrone.streamoff()
myDrone.streamon()

# Load custom face detection model (YOLOv11n)
model = YOLO("yolo11n-face.pt")

# Video capture from Tello stream
cap = cv2.VideoCapture("udp://0.0.0.0:11111", cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 15)

# Parameters
w, h = 360, 240              # Frame size
fbRange = [15000, 30000]     # Face area range for forward/backward control
pid_yaw = [0.4, 0.4, 0]      # [Kp, Kd, Ki] for yaw
pid_ud = [0.3, 0.3, 0]       # [Kp, Kd, Ki] for up/down
pErrorYaw = 0                # Previous yaw error
pErrorUD = 0                 # Previous up/down error

tracking_enabled = True


# -------------------- FACE DETECTION --------------------
def findFace(img):
    results = model(img, conf=0.5, verbose=False, imgsz=256)

    centers = []
    areas = []

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) // 2)
            cy = int((y1 + y2) // 2)
            area = int((x2 - x1) * (y2 - y1))

            # Draw bounding box, label and center
            cvzone.cornerRect(img, (int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
            cv2.putText(img, f"face {confs[i]:.2f}",
                        (int(x1) + 10, int(y1) + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

            centers.append([cx, cy])
            areas.append(area)

    # Track the largest (closest) face
    if areas:
        i = areas.index(max(areas))
        return img, [centers[i], areas[i]]
    return img, [[0, 0], 0]


# -------------------- TRACKING --------------------
def trackFace(info, frame_width, frame_height, pid_yaw, pid_ud, pErrorYaw, pErrorUD):
    cx, cy = info[0]
    area = info[1]

    fb = ud = yaw = 0

    # Yaw control (keep face centered horizontally)
    errorYaw = cx - frame_width // 2
    yaw = pid_yaw[0] * errorYaw + pid_yaw[1] * (errorYaw - pErrorYaw)
    yaw = int(np.clip(yaw, -60, 60))

    # Up/down control (keep face centered vertically)
    if cy != 0:
        errorUD = (frame_height // 2) - cy
        ud = pid_ud[0] * errorUD + pid_ud[1] * (errorUD - pErrorUD)
        ud = int(np.clip(ud, -40, 40))
    else:
        errorUD = 0

    # Forward/backward control (keep a safe distance)
    if area > fbRange[1]:
        fb = -20
    elif 0 < area < fbRange[0]:
        fb = 20

    # No face detected: hover
    if cx == 0:
        yaw = fb = ud = 0
        errorYaw = errorUD = 0

    return yaw, fb, ud, errorYaw, errorUD


# -------------------- KEYBOARD --------------------
def getKeyboardInput():
    global tracking_enabled

    lr = fb = ud = yv = 0
    speed = 40

    if kp.getKey("LEFT"):  lr = -speed
    if kp.getKey("RIGHT"): lr = speed
    if kp.getKey("UP"):    fb = speed
    if kp.getKey("DOWN"):  fb = -speed
    if kp.getKey("w"):     ud = speed
    if kp.getKey("s"):     ud = -speed
    if kp.getKey("a"):     yv = -speed
    if kp.getKey("d"):     yv = speed
    if kp.getKey("e"):
        myDrone.takeoff()
        time.sleep(0.3)
    if kp.getKey("q"):
        myDrone.land()
        time.sleep(0.3)

    # Toggle tracking on/off
    if kp.getKey("t"):
        tracking_enabled = not tracking_enabled
        print("Tracking:", tracking_enabled)
        time.sleep(0.3)

    return lr, fb, ud, yv


# -------------------- MAIN LOOP --------------------
print("Starting Face Tracking...")
print("Controls: E=Takeoff, Q=Land, T=Toggle Tracking, X=Exit")

while True:
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.resize(img, (w, h))
    img, info = findFace(img)

    lr, fb_manual, ud_manual, yv_manual = getKeyboardInput()

    if tracking_enabled:
        yaw_auto, fb_auto, ud_auto, pErrorYaw, pErrorUD = trackFace(
            info, w, h, pid_yaw, pid_ud, pErrorYaw, pErrorUD
        )
    else:
        yaw_auto = fb_auto = ud_auto = 0
        pErrorYaw = pErrorUD = 0

    # Manual keyboard input overrides auto tracking
    fb = fb_manual if fb_manual != 0 else fb_auto
    yv = yv_manual if yv_manual != 0 else yaw_auto
    ud = ud_manual if ud_manual != 0 else ud_auto

    myDrone.send_rc_control(lr, fb, ud, yv)

    if info[0][0] != 0:
        cv2.putText(img, f"Area: {info[1]}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("Face Tracking", img)

    if cv2.waitKey(1) & 0xFF == ord("x"):
        print("Landing and exiting...")
        myDrone.land()
        break

# -------------------- CLEANUP --------------------
cap.release()
cv2.destroyAllWindows()
myDrone.streamoff()
myDrone.end()
