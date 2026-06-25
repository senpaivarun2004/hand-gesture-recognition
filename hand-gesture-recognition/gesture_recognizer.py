import math


TIP_IDS = [4, 8, 12, 16, 20]
PIP_IDS = [3, 6, 10, 14, 18]
MCP_IDS = [2, 5, 9, 13, 17]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


def distance(p1, p2):
    return math.sqrt((p1[1] - p2[1]) ** 2 + (p1[0] - p2[0]) ** 2)


def is_finger_extended(hand_lms, finger_idx, is_thumb=False):
    if is_thumb:
        tip = hand_lms[TIP_IDS[0]]
        ip = hand_lms[3]
        mcp = hand_lms[MCP_IDS[0]]
        return distance(tip, mcp) > distance(ip, mcp) * 1.3
    else:
        tip = hand_lms[TIP_IDS[finger_idx]]
        pip = hand_lms[PIP_IDS[finger_idx]]
        return tip[2] < pip[2]


def get_extended_fingers(hand_data):
    landmarks_xy = [(lm[1], lm[2]) for lm in hand_data["landmarks"]]
    landmarks_z = [lm[3] for lm in hand_data["landmarks"]]
    hand_lms = list(zip(landmarks_xy, landmarks_z))
    hand_lms = [(x, y, z) for (x, y), z in hand_lms]

    fingers = []
    for i in range(5):
        extended = is_finger_extended(hand_lms, i, is_thumb=(i == 0))
        fingers.append(extended)
    return fingers


def recognize_gesture(hand_data):
    fingers = get_extended_fingers(hand_data)
    landmarks_xy = [(lm[1], lm[2]) for lm in hand_data["landmarks"]]
    label = hand_data.get("label")

    extended_count = sum(fingers)

    if fingers == [True, True, True, True, True]:
        return "OPEN_PALM"

    if fingers == [True, False, False, False, False]:
        return "THUMBS_UP"

    if fingers == [False, False, False, False, False]:
        return "FIST"

    if fingers == [False, True, True, False, False]:
        return "PEACE"

    if fingers == [True, False, False, False, True]:
        return "ROCK_ON"

    if fingers == [False, True, False, False, False]:
        return "POINTING"

    if fingers == [False, True, True, True, False]:
        thumb_tip = landmarks_xy[4]
        pinky_tip = landmarks_xy[20]
        if distance(thumb_tip, pinky_tip) < 50:
            return "OK"

    if fingers == [True, True, True, False, False]:
        return "THREE"

    if fingers == [False, False, True, False, False]:
        return "MIDDLE_FINGER"

    if extended_count == 0:
        return "FIST"
    if extended_count >= 4:
        return "OPEN_PALM"

    return f"UNKNOWN ({extended_count} fingers)"
