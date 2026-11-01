# Tello Gesture and Face Control

Real-time hand gesture control and face-tracking for a DJI Tello drone using YOLOv11n models trained on custom labeled datasets.

## Features

- **Gesture control:** 5 hand gestures (stop, left, right, up, down) move the drone.
- **Face tracking:** The drone follows a face using PD control for yaw, height and distance.
- **Manual override:** Keyboard control is always active for safety.

## Repository Structure

```
├── face_tracking.py          # Face-following script
├── gesture_control.py        # Hand gesture control script
├── KeyPressModule.py         # Keyboard input helper (pygame)
├── yolo11n-face.pt           # Trained face detection model
├── yolo11n-handgestures.pt   # Trained hand gesture model
└── training/
    └── train_yolo11n.ipynb   # Colab notebook used to train both models
```

## Requirements

```bash
pip install djitellopy opencv-python ultralytics cvzone pygame numpy
```

## How to Run

1. Turn on the Tello and connect your PC to its Wi-Fi.
2. Run one of the scripts:

```bash
python face_tracking.py
python gesture_control.py
```

## Controls

| Key | Action |
|-----|--------|
| E | Take off |
| Q | Land |
| Arrow keys | Left, right, forward, backward |
| W / S | Up / down |
| A / D | Rotate left / right |
| T | Toggle face tracking (face_tracking.py only) |
| X | Land and exit |

## Training

Both models were trained with `training/train_yolo11n.ipynb` on Google Colab.
The datasets are not shared for privacy reasons.

## Acknowledgements

Adapted from the [Drone Programming Course](https://www.computervision.zone/courses/drone-programming-course/) by CV Zone.
