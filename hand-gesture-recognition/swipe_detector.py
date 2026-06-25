import time


class SwipeDetector:
    """Detects left/right swipe gestures by tracking hand centroid movement.

    Tracks the palm centroid (average of wrist + MCP joints) across recent
    frames. A swipe is registered when the hand moves horizontally beyond a
    threshold, with horizontal movement dominating vertical movement.
    """

    def __init__(
        self,
        swipe_threshold=80,
        ratio_threshold=1.5,
        cooldown=0.8,
        history_size=8,
        min_frames=4,
    ):
        self.swipe_threshold = swipe_threshold
        self.ratio_threshold = ratio_threshold
        self.cooldown = cooldown
        self.history_size = history_size
        self.min_frames = min_frames

        self._positions = []          # list of (x, y, timestamp)
        self._last_swipe_time = 0.0
        self._direction = None        # "LEFT" or "RIGHT" or None
        self._just_detected = False

    @property
    def direction(self):
        return self._direction

    @property
    def just_detected(self):
        return self._just_detected

    @property
    def last_swipe_time(self):
        return self._last_swipe_time

    def update(self, landmarks):
        """Process hand landmarks from one frame.

        Args:
            landmarks: list of (id, x, y, z) tuples from HandTracker.

        Returns:
            "LEFT", "RIGHT", or None if no swipe detected this update.
        """
        centroid = self._centroid(landmarks)
        if centroid is None:
            self._just_detected = False
            return self._direction

        x, y = centroid
        now = time.time()

        self._positions.append((x, y, now))
        if len(self._positions) > self.history_size:
            self._positions.pop(0)

        # Respect cooldown between swipes
        if now - self._last_swipe_time < self.cooldown:
            self._just_detected = False
            return self._direction

        if len(self._positions) < self.min_frames:
            self._just_detected = False
            return self._direction

        first_x = self._positions[0][0]
        last_x = self._positions[-1][0]
        first_y = self._positions[0][1]
        last_y = self._positions[-1][1]

        dx = last_x - first_x
        dy = last_y - first_y

        if abs(dx) > abs(dy) * self.ratio_threshold and abs(dx) > self.swipe_threshold:
            self._direction = "RIGHT" if dx > 0 else "LEFT"
            self._last_swipe_time = now
            self._just_detected = True
            # Reset history to prevent immediate re-trigger
            self._positions = [(x, y, now)]
        else:
            self._just_detected = False

        return self._direction

    @staticmethod
    def _centroid(landmarks):
        """Calculate palm centroid from wrist (0) and MCP joints (5, 9, 13, 17)."""
        indices = {0, 5, 9, 13, 17}
        sx = sy = count = 0
        for lm in landmarks:
            idx, x, y, _ = lm
            if idx in indices:
                sx += x
                sy += y
                count += 1
        if count == 0:
            return None
        return (sx / count, sy / count)
