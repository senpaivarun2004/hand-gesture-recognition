"""
train_model.py — Generate synthetic hand-landmark data and train an ML gesture classifier.

Produces ``gesture_model.joblib`` consumed by ``ml_recognizer.py``.

Strategy
--------
We generate random landmark configurations by starting from a relaxed "open palm"
template, then independently randomising the extension state of each finger.
The existing *rule-based* recognizer labels each sample so the ML model learns
the same decision boundaries but with smoother, probabilistic output.

Usage
-----
    python train_model.py          # train & save model (default)
    python train_model.py --plot   # show feature-importance chart
"""

import argparse
import os
import pickle
import random as rnd
import sys
from copy import deepcopy

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Ensure the project root is on sys.path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gesture_recognizer import recognize_gesture  # noqa: E402
from ml_recognizer import (                       # noqa: E402
    CLASS_TO_IDX,
    GESTURE_CLASSES,
    NUM_CLASSES,
    extract_features,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_SAMPLES_PER_GESTURE = 600   # per gesture for training
NOISE_SAMPLES = 400             # additional random (UNKNOWN) samples
RANDOM_SEED = 42

rnd.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

MODEL_OUTPUT = os.path.join(os.path.dirname(__file__), "gesture_model.joblib")

# A relaxed "open palm" template (id, x, y, z) — values are roughly
# proportional to what MediaPipe returns for a centred right hand.
# fmt: off
OPEN_PALM_TEMPLATE: list = [
    (0, 300, 400, 0.0),       # WRIST
    (1, 310, 340, 0.01),      # THUMB_CMC
    (2, 345, 335, 0.02),      # THUMB_MCP
    (3, 365, 350, 0.03),      # THUMB_IP
    (4, 390, 360, 0.04),      # THUMB_TIP
    (5, 310, 310, 0.02),      # INDEX_MCP
    (6, 320, 270, 0.04),      # INDEX_PIP
    (7, 325, 240, 0.05),      # INDEX_DIP
    (8, 330, 210, 0.06),      # INDEX_TIP
    (9, 300, 300, 0.02),     # MIDDLE_MCP
    (10, 300, 255, 0.04),    # MIDDLE_PIP
    (11, 300, 220, 0.05),    # MIDDLE_DIP
    (12, 300, 185, 0.06),    # MIDDLE_TIP
    (13, 290, 310, 0.02),    # RING_MCP
    (14, 280, 270, 0.04),    # RING_PIP
    (15, 275, 240, 0.05),    # RING_DIP
    (16, 270, 210, 0.06),    # RING_TIP
    (17, 280, 320, 0.02),    # PINKY_MCP
    (18, 265, 285, 0.04),    # PINKY_PIP
    (19, 260, 260, 0.05),    # PINKY_DIP
    (20, 255, 240, 0.06),    # PINKY_TIP
]
# fmt: on

TIP_IDS = [4, 8, 12, 16, 20]
PIP_IDS = [3, 6, 10, 14, 18]
MCP_IDS = [2, 5, 9, 13, 17]

# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------


def _perturb(template, xy_noise=6.0, z_noise=0.012):
    """Return a copy of *template* with random noise added."""
    out = []
    for idx, x, y, z in template:
        out.append((
            idx,
            x + rnd.uniform(-xy_noise, xy_noise),
            y + rnd.uniform(-xy_noise, xy_noise),
            z + rnd.uniform(-z_noise, z_noise),
        ))
    return out


def _set_finger_curled(lms, finger_idx, curled=True, thumb_extra_curl=False):
    """Adjust landmark positions to simulate curled/extended finger.

    Landmarks are stored as (id, x, y, z) tuples.

    For non-thumb fingers (1-4):
      - Extended → TIP.z < PIP.z
      - Curled   → TIP.z > PIP.z

    For thumb:
      - Extended → TIP farther from MCP than IP is.
      - Curled   → TIP closer to MCP.
    """
    if finger_idx == 0:  # thumb — distance heuristic
        _id0, mcp_x, mcp_y, mcp_z = lms[MCP_IDS[0]]
        _id1, ip_x,  ip_y,  ip_z  = lms[3]
        _id2, t_x,   t_y,   t_z   = lms[TIP_IDS[0]]

        if curled:
            # Pull thumb tip towards MCP (tucked in)
            lms[TIP_IDS[0]] = (
                _id2,
                t_x + (mcp_x - t_x) * 0.5,
                t_y + (mcp_y - t_y) * 0.5,
                t_z - 0.01,
            )
        else:
            # Push thumb tip away from MCP (extended out)
            lms[TIP_IDS[0]] = (
                _id2,
                t_x + (t_x - mcp_x) * 0.3,
                t_y + (t_y - mcp_y) * 0.3,
                t_z + 0.01,
            )
        return

    tip_idx = TIP_IDS[finger_idx]
    pip_idx = PIP_IDS[finger_idx]

    _id_tip, tip_x, tip_y, tip_z = lms[tip_idx]
    _id_pip, _px,   _py,   pip_z = lms[pip_idx]

    if curled:
        # TIP closer to camera than PIP → z > pip_z
        lms[tip_idx] = (_id_tip, tip_x, tip_y, pip_z + rnd.uniform(0.02, 0.07))
    else:
        # TIP farther from camera → z < pip_z
        lms[tip_idx] = (_id_tip, tip_x, tip_y, pip_z - rnd.uniform(0.02, 0.07))


def _set_finger_pattern(template, pattern):
    """Apply a 5-element [thumb, index, middle, ring, pinky] extended pattern."""
    lms = list(template)
    for i, extended in enumerate(pattern):
        _set_finger_curled(lms, i, curled=not extended)
    return lms


def _classify_hand_data(lms_list):
    """Build a ``hand_data`` dict and run the rule-based recognizer."""
    hand_data = {"landmarks": lms_list, "label": "Right"}
    return recognize_gesture(hand_data)


# ---------------------------------------------------------------------------
# Gesture patterns (mirroring gesture_recognizer.py)
# ---------------------------------------------------------------------------

GESTURE_PATTERNS = {
    "OPEN_PALM":      [True, True, True, True, True],
    "FIST":           [False, False, False, False, False],
    "THUMBS_UP":      [True, False, False, False, False],
    "PEACE":          [False, True, True, False, False],
    "POINTING":       [False, True, False, False, False],
    "ROCK_ON":        [True, False, False, False, True],
    "OK":             [False, True, True, True, False],
    "THREE":          [True, True, True, False, False],
    "MIDDLE_FINGER":  [False, False, True, False, False],
}

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_dataset(samples_per_gesture=NUM_SAMPLES_PER_GESTURE,
                     noise_samples=NOISE_SAMPLES):
    """Generate labelled synthetic dataset."""
    features, labels = [], []

    # --- Known gestures ---
    gesture_names = list(GESTURE_PATTERNS.keys())
    for gesture_name in gesture_names:
        pattern = GESTURE_PATTERNS[gesture_name]
        for _ in range(samples_per_gesture):
            # Slightly randomise the open-palm template first
            template = _perturb(OPEN_PALM_TEMPLATE,
                                xy_noise=4.0, z_noise=0.008)
            lms = _set_finger_pattern(template, pattern)
            # Add a second round of perturbation (after finger posing)
            lms = _perturb(lms, xy_noise=3.0, z_noise=0.006)

            # Verify with rule-based — skip if mismatch (shouldn't happen
            # with clean data, but protects against degenerate samples)
            label = _classify_hand_data(lms)

            # For OK sign we must also enforce the thumb-pinky distance
            if gesture_name == "OK":
                thumb_id, thumb_x, thumb_y, thumb_z = lms[4]
                pinky_id, pinky_x, pinky_y, pinky_z = lms[20]
                dx = thumb_x - pinky_x
                dy = thumb_y - pinky_y
                dist = np.sqrt(dx * dx + dy * dy)
                if dist > 45:  # Approximate OK threshold
                    # Pull thumb and pinky closer together
                    mx = (thumb_x + pinky_x) / 2
                    my = (thumb_y + pinky_y) / 2
                    lms[4] = (thumb_id, mx - 15, my + rnd.uniform(-3, 3), thumb_z)
                    lms[20] = (pinky_id, mx + 15, my + rnd.uniform(-3, 3), pinky_z)

            label = _classify_hand_data(lms)

            if label == gesture_name or (gesture_name == "OK" and "OK" in label):
                feat = extract_features(lms)
                features.append(feat)
                labels.append(CLASS_TO_IDX[gesture_name])

    # --- Noise / Unknown samples ---
    for _ in range(noise_samples):
        template = _perturb(OPEN_PALM_TEMPLATE,
                            xy_noise=20.0, z_noise=0.03)
        # Random finger pattern
        random_pattern = [rnd.random() > 0.5 for _ in range(5)]
        lms = _set_finger_pattern(template, random_pattern)
        lms = _perturb(lms, xy_noise=10.0, z_noise=0.02)
        label = _classify_hand_data(lms)

        # Only include if it's unknown or we want more variety
        if "UNKNOWN" in label:
            feat = extract_features(lms)
            # Assign to a random known class as "noise" to reduce overconfidence
            features.append(feat)
            labels.append(rnd.randint(0, NUM_CLASSES - 1))

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train():
    print("Generating synthetic dataset…")
    X, y = generate_dataset()
    print(f"  Total samples: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y,
    )

    print("Training RandomForest classifier…")
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=0,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = (y_pred == y_test).mean()
    print(f"\nTest accuracy: {acc:.3f}\n")
    print(classification_report(
        y_test, y_pred,
        target_names=GESTURE_CLASSES,
        zero_division=0,
    ))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    # Save
    os.makedirs(os.path.dirname(MODEL_OUTPUT) or ".", exist_ok=True)
    with open(MODEL_OUTPUT, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nModel saved to {MODEL_OUTPUT}")

    # Quick sanity check
    _sanity_check(clf)


def _sanity_check(clf):
    """Verify the model can classify a template for each gesture."""
    print("\nSanity check:")
    for name, pattern in GESTURE_PATTERNS.items():
        lms = _set_finger_pattern(OPEN_PALM_TEMPLATE, pattern)
        feat = extract_features(lms).reshape(1, -1)
        pred = clf.predict(feat)[0]
        prob = clf.predict_proba(feat).max()
        print(f"  {name:20s} → {GESTURE_CLASSES[pred]:20s}  (confidence: {prob:.3f})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML gesture classifier")
    parser.add_argument("--plot", action="store_true",
                        help="Show feature-importance chart (requires matplotlib)")
    args = parser.parse_args()

    train()

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            with open(MODEL_OUTPUT, "rb") as f:
                model = pickle.load(f)
            importances = model.feature_importances_
            plt.figure(figsize=(10, 4))
            plt.bar(range(len(importances)), importances)
            plt.title("Random Forest Feature Importances (63 normalized landmarks)")
            plt.xlabel("Feature index")
            plt.ylabel("Importance")
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib not installed, skipping plot.")
