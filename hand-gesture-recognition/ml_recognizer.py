"""
ml_recognizer.py — Machine-learning gesture recognizer.

Uses a scikit-learn RandomForestClassifier trained on normalized 21-landmark
features (63 dimensions) to classify hand gestures with confidence scores.

Falls back to the rule-based recognizer when ML confidence is low.
"""

import os
import pickle
from typing import Optional

import numpy as np

from gesture_recognizer import recognize_gesture

# ---------------------------------------------------------------------------
# Landmark indices used for normalisation
# ---------------------------------------------------------------------------
WRIST = 0
MIDDLE_MCP = 9  # used as the reference distance for scale normalisation

# Gesture label ↔ integer mapping
GESTURE_CLASSES = [
    "OPEN_PALM",
    "FIST",
    "THUMBS_UP",
    "PEACE",
    "POINTING",
    "ROCK_ON",
    "OK",
    "THREE",
    "MIDDLE_FINGER",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(GESTURE_CLASSES)}
NUM_CLASSES = len(GESTURE_CLASSES)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "gesture_model.joblib")


# ---------------------------------------------------------------------------
# Feature extraction — normalise landmarks to be invariant to position & size
# ---------------------------------------------------------------------------


def extract_features(landmarks):
    """Convert a list of (id, x, y, z) landmarks into a normalised 63-vector.

    Normalisation steps
    -------------------
    1. Centre — subtract wrist (x, y) so the hand position is irrelevant.
    2. Scale  — divide by the distance from wrist → middle-finger MCP so that
               hand size & distance-from-camera are factored out.
    3. Z is left as MediaPipe-relative (already wrist-relative).
    """
    # Build arrays
    xs = np.array([lm[1] for lm in landmarks], dtype=np.float32)
    ys = np.array([lm[2] for lm in landmarks], dtype=np.float32)
    zs = np.array([lm[3] for lm in landmarks], dtype=np.float32)

    # 1. Centre on wrist
    xs -= xs[WRIST]
    ys -= ys[WRIST]

    # 2. Scale by wrist → middle-MCP distance
    dx = xs[MIDDLE_MCP] - xs[WRIST]
    dy = ys[MIDDLE_MCP] - ys[WRIST]
    scale = max(np.sqrt(dx * dx + dy * dy), 1.0)
    xs /= scale
    ys /= scale

    # 3. Flatten into a single 63-element vector
    return np.concatenate([xs, ys, zs])  # shape (63,)


# ---------------------------------------------------------------------------
# ML recogniser
# ---------------------------------------------------------------------------


class MLGestureRecognizer:
    """Wraps a trained sklearn classifier for gesture recognition.

    Usage
    -----
    >>> recognizer = MLGestureRecognizer()
    >>> gesture, confidence = recognizer.predict(hand_landmarks)
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self._model: Optional["RandomForestClassifier"] = None
        self._model_path = model_path
        self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, landmarks) -> tuple:
        """Return (gesture_name, confidence) or (rule_based_gesture, 0.0)."""
        if self._model is None:
            return self._fallback(landmarks)

        features = extract_features(landmarks).reshape(1, -1)

        probs = self._model.predict_proba(features)[0]  # shape (NUM_CLASSES,)
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])

        if confidence >= self._min_confidence():
            return GESTURE_CLASSES[best_idx], confidence

        # Low confidence → fall back to rules
        return self._fallback(landmarks)

    def predict_with_all_probs(self, landmarks):
        """Return (gesture_name, confidence, {gesture: prob, ...})."""
        if self._model is None:
            rule = self._fallback(landmarks)
            return rule, 0.0, {rule: 1.0}

        features = extract_features(landmarks).reshape(1, -1)
        probs = self._model.predict_proba(features)[0]
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        best_gesture = GESTURE_CLASSES[best_idx]

        prob_map = {GESTURE_CLASSES[i]: float(probs[i]) for i in range(NUM_CLASSES)}

        if confidence >= self._min_confidence():
            return best_gesture, confidence, prob_map
        rule = self._fallback(landmarks)
        return rule, confidence, prob_map

    @property
    def available(self) -> bool:
        return self._model is not None

    def reload(self):
        self._load_model()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_model(self):
        if not os.path.isfile(self._model_path):
            print(
                f"[ml_recognizer] Model not found at {self._model_path}. "
                f"Run train_model.py first to generate it."
            )
            self._model = None
            return
        try:
            with open(self._model_path, "rb") as f:
                self._model = pickle.load(f)
            print(f"[ml_recognizer] Loaded model from {self._model_path}")
        except Exception as exc:
            print(f"[ml_recognizer] Failed to load model: {exc}")
            self._model = None

    @staticmethod
    def _min_confidence():
        return float(
            os.environ.get("ML_CONFIDENCE_THRESHOLD", "0.55")
        )

    @staticmethod
    def _fallback(landmarks):
        # Build a minimal hand_data dict for the rule-based recognizer
        hand_data = {"landmarks": landmarks, "label": ""}
        return recognize_gesture(hand_data)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_global_recognizer: Optional[MLGestureRecognizer] = None


def get_global_recognizer() -> MLGestureRecognizer:
    """Return (and cache) a singleton MLGestureRecognizer."""
    global _global_recognizer
    if _global_recognizer is None:
        _global_recognizer = MLGestureRecognizer()
    return _global_recognizer


def ml_recognize_gesture(landmarks) -> tuple:
    """Convenience: returns (gesture_name, confidence)."""
    return get_global_recognizer().predict(landmarks)
