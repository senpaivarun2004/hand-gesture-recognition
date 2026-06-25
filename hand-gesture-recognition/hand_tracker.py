import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
    drawing_utils,
)
from mediapipe.tasks.python.vision.drawing_utils import DrawingSpec

# Path to the hand_landmarker model file (expected next to this script)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


class HandTracker:
    def __init__(self, max_hands=2, detection_confidence=0.7, tracking_confidence=0.5):
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"Hand landmarker model not found at: {_MODEL_PATH}\n"
                "Download it from: https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self.results = None
        self._frame_timestamp_ms = 0

        # Drawing specs
        self._landmark_style = DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        self._connection_style = DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1)

    def find_hands(self, frame, draw=True):
        """Detect hands and optionally draw landmarks on the frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        self._frame_timestamp_ms += 33  # ~30 fps
        self.results = self.landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        if self.results and self.results.hand_landmarks and draw:
            for hand_landmarks in self.results.hand_landmarks:
                drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    HandLandmarksConnections.HAND_CONNECTIONS,
                    self._landmark_style,
                    self._connection_style,
                )
        return frame

    def get_landmarks(self, frame):
        """Return a list of hand data dicts compatible with gesture_recognizer.

        Each dict has:
          - "index": hand index
          - "landmarks": list of (idx, x_px, y_px, z) tuples
          - "label": "Left" or "Right"
        """
        h, w, _ = frame.shape
        landmarks = []

        if not self.results or not self.results.hand_landmarks:
            return landmarks

        for hand_index, hand_lms in enumerate(self.results.hand_landmarks):
            hand_data = {"index": hand_index, "landmarks": [], "label": None}

            # Handedness
            if self.results.handedness and hand_index < len(self.results.handedness):
                hand_data["label"] = self.results.handedness[hand_index][0].category_name

            for idx, lm in enumerate(hand_lms):
                x = int(lm.x * w)
                y = int(lm.y * h)
                hand_data["landmarks"].append((idx, x, y, lm.z))

            landmarks.append(hand_data)

        return landmarks

    def release(self):
        self.landmarker.close()
