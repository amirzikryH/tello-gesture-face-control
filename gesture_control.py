import cv2
import time
import threading
import queue
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

# Load custom hand gesture model (YOLOv11n)
gesture_model = YOLO("yolo11n-handgestures.pt")

# Class names (must match the order in data.yaml)
class_names = ["stop", "left", "right", "up", "down"]

# Video capture from Tello stream
cap = cv2.VideoCapture("udp://0.0.0.0:11111", cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 15)

print("Waiting for stream...")
for _ in range(10):
    cap.read()
print("Stream ready")

# Grab frames in a background thread so only the latest frame is used
frame_queue = queue.Queue(maxsize=1)
stop_thread = False


def grab_frames():
    while not stop_thread:
        ret, frame = cap.read()
        if ret:
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put(frame)


grab_thread = threading.Thread(target=grab_frames)
grab_thread.start()

w, h = 360, 240


# -------------------- GESTURE DETECTION --------------------
def detect_gesture(img):
    """Returns the annotated image and the detected class ID (-1 if none)."""
    results = gesture_model(img, conf=0.6, verbose=False, imgsz=256)

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        # Use the highest-confidence detection
        box = results[0].boxes.xyxy[0].cpu().numpy()
        class_id = int(results[0].boxes.cls[0].cpu().numpy())
        conf = results[0].boxes.conf[0].cpu().numpy()

        x1, y1, x2, y2 = box
        cvzone.cornerRect(img, (int(x1), int(y1), int(x2 - x1), int(y2 - y1)), t=3)
        cv2.putText(img, f"{class_names[class_id]} {conf:.2f}",
                    (int(x1) + 10, int(y1) + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return img, class_id

    return img, -1


# -------------------- KEYBOARD (manual override) --------------------
def getKeyboardInput():
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

    return lr, fb, ud, yv


# -------------------- MAIN LOOP --------------------
print("Starting Gesture Control...")
print("Gestures: stop, left, right, up, down")
print("Manual: Arrows/WASD, E=Takeoff, Q=Land, X or ESC=Exit")

frame_counter = 0
gesture_id = -1
gesture_speed = 25

while True:
    try:
        img = frame_queue.get(timeout=1.0)
    except queue.Empty:
        continue

    img = cv2.resize(img, (w, h))

    # Run detection every 2nd frame to save compute
    frame_counter += 1
    if frame_counter % 2 == 0:
        img, gesture_id = detect_gesture(img)

    lr_k, fb_k, ud_k, yv_k = getKeyboardInput()

    # Map gesture to drone command (mirrored left/right since the drone faces the user)
    lr_g = fb_g = ud_g = yv_g = 0
    if gesture_id == 1:    # left  -> drone moves right
        lr_g = gesture_speed
    elif gesture_id == 2:  # right -> drone moves left
        lr_g = -gesture_speed
    elif gesture_id == 3:  # up
        ud_g = gesture_speed
    elif gesture_id == 4:  # down
        ud_g = -gesture_speed
    # stop (0) or no gesture (-1): hover

    # Manual keyboard input overrides gestures
    lr = lr_k if lr_k != 0 else lr_g
    fb = fb_k if fb_k != 0 else fb_g
    ud = ud_k if ud_k != 0 else ud_g
    yv = yv_k if yv_k != 0 else yv_g

    myDrone.send_rc_control(lr, fb, ud, yv)

    # On-screen info
    cv2.putText(img, "GESTURE CONTROL", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if gesture_id != -1:
        cv2.putText(img, f"Command: {class_names[gesture_id]}", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(img, f"LR:{lr} FB:{fb} UD:{ud} YV:{yv}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    cv2.imshow("Gesture Control", img)

    if cv2.waitKey(1) & 0xFF == 27 or kp.getKey("x"):
        print("Landing...")
        myDrone.land()
        break

# -------------------- CLEANUP --------------------
stop_thread = True
grab_thread.join()
cap.release()
cv2.destroyAllWindows()
myDrone.streamoff()
myDrone.end()
