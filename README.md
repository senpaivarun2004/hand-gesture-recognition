### Hand-Gesture-Recognition
# 📸 Hand Gesture Recognition System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python Version">
  <img src="https://img.shields.io/badge/MediaPipe-0.10%2B-teal?style=for-the-badge&logo=google&logoColor=white" alt="MediaPipe">
  <img src="https://img.shields.io/badge/OpenCV-4.8%2B-green?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/PyQt-5-orange?style=for-the-badge&logo=qt&logoColor=white" alt="PyQt5">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License MIT">
</p>

A state-of-the-art, real-time hand gesture recognition system built with **MediaPipe** and **OpenCV**. It includes both a rule-based engine and a machine learning classifier, alongside a polished **PyQt5 dashboard** for customization, visualization, and profiling.

---

## ✨ Features

- 🖐️ **21-Point Landmark Tracking:** Real-time 3D hand mesh detection powered by Google's MediaPipe.
- ⚙️ **Dual Classification Engines:**
  - **Rule-Based Recognizer:** High-speed heuristic classification based on relative joint angles and finger extensions.
  - **Machine Learning Recognizer:** Trained `RandomForestClassifier` (Scikit-Learn) with normalized, position/scale-invariant feature extraction.
- 🖥️ **Interactive PyQt5 Dashboard:**
  - Live video preview with tracking overlay.
  - Interactive adjustments (confidence thresholds, max hands, FPS toggles).
  - Dynamic profile manager (Save, Load, and Reset presets).
  - Live scrollable gesture & swipe event logger.
- 📱 **Swipe Presentation Control:** Detects fast hand swiping motions (Left/Right) with animated overlays, ideal for controlling presentations or media players.
- 🛠️ **Custom Training Toolkit:** Easily gather custom gesture data and retrain the classifier with built-in scripts.

---

## 🎨 Supported Gestures

| Gesture | Label | Visual Pose | Primary Action / Use Case |
| :--- | :---: | :---: | :--- |
| **Open Palm** | `OPEN_PALM` | 🖐️ | Stop presentation / Pause video / Greeting |
| **Fist** | `FIST` | ✊ | Hold / Grab / Punch |
| **Thumbs Up** | `THUMBS_UP` | 👍 | Confirm / Approve / Like |
| **Peace Sign** | `PEACE` | ✌️ | Take photo / Trigger action / Number 2 |
| **OK Sign** | `OK` | 👌 | Select / OK / Confirm |
| **Pointing** | `POINTING` | ☝️ | Air-mouse cursor / Draw / Laser pointer |
| **Rock On** | `ROCK_ON` | 🤘 | Skip track / Rock mode |
| **Three** | `THREE` | 🤟 | Number 3 / Special action |
| **Middle Finger**| `MIDDLE_FINGER`| 🖕 | Not nice! |

---

## 🏗️ Architecture & Processing Pipeline

```
       Webcam Video Stream
               │
               ▼
       ┌───────────────┐
       │  OpenCV Frame │
       └───────┬───────┘
               │ (Flip/Color Convert)
               ▼
       ┌───────────────┐
       │ MediaPipe UI  ├─► Extracts 21 3D Landmarks
       └───────┬───────┘
               │
               ▼
     ┌───────────────────┐
     │ Feature Extractor │ (Wrist centering & MCP scale normalisation)
     └─────────┬─────────┘
               │
               ├────────────────────────┐
               ▼                        ▼
      ┌──────────────────┐    ┌────────────────────┐
      │ Rule-Based Model │    │ Scikit-Learn ML    │
      │  (Joint angles)  │    │ (Random Forest)    │
      └────────┬─────────┘    └─────────┬──────────┘
               │                        │ (If confidence > threshold)
               └───────────┬────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Gesture Output  │ (Render Overlay, Swipe detection)
                  └─────────────────┘
```

---

## 📂 Project Structure

- [main.py](file:///c:/Users/senpq/Projects/hand-gesture-recognition/main.py) — Command-line launcher for webcam loop and basic overlays.
- [dashboard.py](file:///c:/Users/senpq/Projects/hand-gesture-recognition/dashboard.py) — Polished PyQt5 GUI Application with complete controls.
- [hand_tracker.py](file:///c:/Users/senpq/Projects/hand-gesture-recognition/hand_tracker.py) — Encapsulated wrapper for MediaPipe Hands API.
- [gesture_recognizer.py](file:///c:/Users/senpq/Projects/hand-gesture-recognition/gesture_recognizer.py) — Algorithmic rule-based detector (finger states & orientations).
- [ml_recognizer.py](file:///c:/Users/senpq/Projects/hand-gesture-recognition/ml_recognizer.py) — Feature extraction and RandomForest predictor.
- [train_model.py](file:///c:/Users/senpq/Projects/hand-gesture-recognition/train_model.py) — Utility script for training the classifier.
- [swipe_detector.py](file:///c:/Users/senpq/Projects/hand-gesture-recognition/swipe_detector.py) — Detects swipes based on hand centroid speed and direction.
- `gesture_model.joblib` — Serialized pre-trained Random Forest model.
- `hand_landmarker.task` — MediaPipe Hand Landmarker model file.
- `profiles/` — Saved configuration profiles JSON files.

---

## 🚀 Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.8 or newer** installed.

### 2. Clone the Repository
```bash
git clone https://github.com/yourusername/hand-gesture-recognition.git
cd hand-gesture-recognition
```

### 3. Create & Activate Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🎮 How to Run

### Option A: The Full GUI Dashboard (Recommended)
Run the feature-rich Qt GUI console:
```bash
python dashboard.py
```
> **Tip:** In the Dashboard, you can save custom configurations for different environments, adjust detection sensitivity, and view the live event log.

### Option B: Fast Command Line Preview
Run the lightweight preview overlay:
```bash
python main.py
```
- Press `q` in the video window to quit.

---

## 🤖 Training the ML Model

If you wish to retrain the gesture recognition model or add new gestures:

1. Use feature extraction in `ml_recognizer.py` or collect coordinates.
2. Run the training script:
   ```bash
   python train_model.py
   ```
3. The script outputs a new `gesture_model.joblib` which is automatically picked up by the application.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
