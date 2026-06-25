import time

import cv2

from hand_tracker import HandTracker
from gesture_recognizer import recognize_gesture
from swipe_detector import SwipeDetector

GESTURE_INFO = {
    "OPEN_PALM": ("Stop / Hi!", (0, 255, 0)),
    "FIST": ("Punch / Hold", (0, 0, 255)),
    "THUMBS_UP": ("Like / Confirm", (255, 255, 0)),
    "PEACE": ("Victory / V-sign", (255, 0, 255)),
    "OK": ("OK sign", (0, 255, 255)),
    "POINTING": ("Point / Select", (255, 128, 0)),
    "ROCK_ON": ("Rock On! \\m/", (128, 0, 255)),
    "THREE": ("Number 3", (0, 200, 200)),
    "MIDDLE_FINGER": ("Not nice!", (0, 0, 200)),
}

SWIPE_INFO = {
    "LEFT": ("◀◀  SWIPE LEFT  ◀◀", (255, 165, 0)),
    "RIGHT": ("▶▶  SWIPE RIGHT  ▶▶", (0, 200, 255)),
}

CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
SWIPE_DISPLAY_SECS = 0.7


def draw_swipe_overlay(frame, swipe_detector):
    """Draw an animated swipe indicator at the top of the frame."""
    if swipe_detector.direction is None:
        return

    elapsed = time.time() - swipe_detector.last_swipe_time
    if elapsed > SWIPE_DISPLAY_SECS:
        return

    direction = swipe_detector.direction
    text, color = SWIPE_INFO.get(direction, ("", (200, 200, 200)))

    # Fade out over the display window
    alpha = max(0.0, 1.0 - elapsed / SWIPE_DISPLAY_SECS)
    faded = tuple(int(c * alpha) for c in color)

    h, w, _ = frame.shape

    # Dark overlay bar at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    # Outer border
    cv2.rectangle(frame, (2, 2), (w - 2, 78), faded, 2)

    # Text centered in the bar
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.3
    thickness = 3
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    tx = (w - tw) // 2
    ty = (80 + th) // 2

    cv2.putText(frame, text, (tx, ty), font, scale, faded, thickness, cv2.LINE_AA)

    return frame


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    tracker = HandTracker()
    swipe_detector = SwipeDetector()

    print("Hand Gesture Recognition is running...")
    print("Press 'q' to quit.")
    print("Swipe left/right with your hand for presentation control!")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape

        frame = tracker.find_hands(frame)

        hands = tracker.get_landmarks(frame)

        total_hands = len(hands)

        for hand_index, hand in enumerate(hands):
            gesture = recognize_gesture(hand)
            info, color = GESTURE_INFO.get(gesture, (gesture, (200, 200, 200)))

            wrist_x = hand["landmarks"][0][1]
            wrist_y = hand["landmarks"][0][2]

            label = hand.get("label", "")
            display_text = f"{gesture}: {info}"

            # ---- Smart label box positioning ----
            BOX_W, BOX_H = 300, 120
            STATUS_BAR_H = 45  # keep clear of the status text at top
            PAD = 8

            # Determine which side of the hand to place the box
            # left hand → box goes to the right; right hand → box goes to the left
            # When only one hand, prefer placing above the wrist
            if total_hands == 1:
                # Center box horizontally over the wrist
                box_x1 = wrist_x - BOX_W // 2
                box_y1 = wrist_y - BOX_H - 10
            else:
                # Two hands: place on opposite sides to avoid overlap
                if label == "Left" or (label == "" and hand_index == 0):
                    box_x1 = wrist_x + 30
                    box_y1 = wrist_y - BOX_H // 2
                else:
                    box_x1 = wrist_x - BOX_W - 30
                    box_y1 = wrist_y - BOX_H // 2

            # Clamp inside frame, keeping clear of the status bar
            box_x1 = max(PAD, min(box_x1, frame_width - BOX_W - PAD))
            # If the original intended position was above the valid area,
            # flip to place the box below the wrist instead
            if box_y1 < STATUS_BAR_H:
                box_y1 = wrist_y + 20
            box_y1 = max(STATUS_BAR_H, min(box_y1, frame_height - BOX_H - PAD))

            box_x2 = box_x1 + BOX_W
            box_y2 = box_y1 + BOX_H

            # ---- Render ----
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), color, 2)

            text_x = box_x1 + 12
            cv2.putText(
                frame,
                display_text,
                (text_x, box_y1 + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"[{label}]",
                (text_x, box_y1 + 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Update swipe detection with the first detected hand
        if hands:
            swipe_detector.update(hands[0]["landmarks"])

        # Draw swipe overlay (before status text so it sits behind it)
        result_frame = draw_swipe_overlay(frame, swipe_detector)
        if result_frame is not None:
            frame = result_frame

        # Status bar — drawn last so it's always on top
        cv2.putText(
            frame,
            "Press 'q' to quit  |  Swipe left/right for slides",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (100, 100, 100),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    tracker.release()
    cv2.destroyAllWindows()
    print("Application closed.")


if __name__ == "__main__":
    main();
