# Hand Gesture Recognition

A real-time hand gesture recognition system using OpenCV and MediaPipe. Detects hand landmarks from a webcam feed, classifies gestures, and overlays the result on the video stream.

## Demo Scenario (For External Mock Presentation)

### The Problem
Traditional human-computer interaction relies on physical controllers, keyboards, or touchscreens. In scenarios like **smart presentations, AR/VR control, assistive technology, or contactless interfaces** (e.g., hospital operating rooms, factory floors), touchless gesture control is far more hygienic, accessible, and intuitive.

### The Solution
This project demonstrates a **real-time hand gesture recognition system** that replaces physical input with natural hand movements:

| Gesture | Recognized As | Real-World Use Case |
|---------|--------------|---------------------|
| Open Palm | Stop / Hi | Smart home: pause music, stop a robot |
| Fist | Punch / Hold | Gaming: punch action, grab an object |
| Thumbs Up | Like / Confirm | Social media: like a post, confirm action |
| Peace Sign | Victory / V | Photo booth: trigger camera shutter |
| OK Sign | OK / Confirm | AR/VR: confirm selection |
| Pointing | Point / Select | Presentation: act as a laser pointer |
| Rock On | Rock On \m/ | Music app: skip track / rock mode |

### Presentation Walkthrough

```
Step 1 – Open the app
  "This is a real-time hand gesture recognition system. 
   It uses a webcam to detect 21 landmark points on your hand 
   and classifies them into meaningful gestures."

Step 2 – Show Open Palm
  "Here you see the open palm gesture. It detects all 5 fingers 
   extended. This could be used as a 'stop' signal or to say 'hi' 
   in a virtual meeting."

Step 3 – Make a Fist
  "When I close my hand into a fist, it detects 0 extended fingers. 
   In a game, this could mean 'punch'. In a robotics scenario, 
   this could mean 'grab'."

Step 4 – Thumbs Up
  "Thumbs up is a universal sign of approval. It detects only 
   the thumb extended. Useful for 'like' buttons or confirming 
   an action without touching a screen."

Step 5 – Peace / Point / OK
  "The system also recognizes the peace sign, pointing finger, 
   OK sign, and more. Each gesture maps to a different action."
```

### How It Works (Technical)

```
Webcam Frame → HandTracker (MediaPipe) → 21 Landmarks → GestureRecognizer → Label + Display
```

1. **Frame Capture** – OpenCV reads frames from the webcam.
2. **Hand Detection** – MediaPipe Hands detects up to 2 hands with 21 landmarks each.
3. **Landmark Extraction** – Each landmark has (x, y, z) coordinates in pixel space.
4. **Gesture Classification** – Rule-based logic checks which fingers are extended using relative positions of fingertips (TIP) vs PIP joints.
5. **Overlay Rendering** – The gesture name, description, and hand label (Left/Right) are displayed on the frame.

### File Structure

```
hand-gesture-recognition/
  main.py                 # Entry point: webcam loop & display
  hand_tracker.py         # MediaPipe wrapper for hand detection
  gesture_recognizer.py   # Gesture classification logic
  requirements.txt        # Python dependencies
  README.md               # This file
```

### Installation

```bash
# 1. Clone or download the project
cd hand-gesture-recognition

# 2. (Recommended) Create a virtual environment
python -m venv venv
.\venv\Scripts\Activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
python main.py
```

Press **`q`** to quit the application.

### Dependencies

- Python 3.8+
- OpenCV (`opencv-python`) – camera access & image processing
- MediaPipe (`mediapipe`) – hand landmark detection
- NumPy (`numpy`) – coordinate calculations

### Key Functions

| File | Function | Purpose |
|------|----------|---------|
| `hand_tracker.py` | `find_hands(frame)` | Detects hands and draws landmarks |
| `hand_tracker.py` | `get_landmarks(frame)` | Returns raw landmark coordinates |
| `gesture_recognizer.py` | `recognize_gesture(hand_data)` | Classifies the hand pose |
| `gesture_recognizer.py` | `get_extended_fingers(hand_data)` | Determines which fingers are up |

### Possible Extensions

- Add swipe gestures (left/right/up/down) using hand trajectory tracking
- Map gestures to keyboard/mouse actions (e.g., volume control, slide navigation)
- Train a machine learning classifier (e.g., TensorFlow) for more complex gestures
- Add gesture-to-speech for accessibility (sign language translation)
- Deploy on Raspberry Pi for embedded contactless kiosk control

### License

MIT
