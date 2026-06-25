"""
Hand Gesture Recognition — PyQt5 Dashboard

Provides a GUI with:
  - Live camera preview with gesture overlays
  - Settings panel (thresholds, toggles, etc.)
  - Gesture history log
  - Profile save / load / reset
"""

import json
import os
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QCloseEvent, QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gesture_recognizer import recognize_gesture
from hand_tracker import HandTracker
from main import GESTURE_INFO, SWIPE_INFO, draw_swipe_overlay
from swipe_detector import SwipeDetector

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROFILES_DIR = "profiles"
os.makedirs(PROFILES_DIR, exist_ok=True)

DEFAULT_PROFILE = {
    "swipe_threshold": 80,
    "swipe_cooldown": 0.8,
    "max_hands": 2,
    "detection_confidence": 0.7,
    "tracking_confidence": 0.5,
    "show_landmarks": True,
    "show_gesture_labels": True,
    "show_fps": True,
    "camera_index": 0,
    "capture_width": 1280,
    "capture_height": 720,
}

MAX_HISTORY_ROWS = 500

# ---------------------------------------------------------------------------
# CameraWorker — runs the camera loop in a background thread
# ---------------------------------------------------------------------------


class CameraWorker(QObject):
    """Processes webcam frames in a background QThread."""

    frame_ready = pyqtSignal(QImage)
    gesture_detected = pyqtSignal(str, str, str)  # timestamp, gesture, hand_label
    swipe_detected = pyqtSignal(str)  # "LEFT" or "RIGHT"
    fps_updated = pyqtSignal(float)
    error_occurred = pyqtSignal(str)

    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._running = False

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @pyqtSlot()
    def start(self):
        """Main loop — run in the worker thread via QThread.started."""
        self._running = True

        cap = cv2.VideoCapture(self._settings["camera_index"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._settings["capture_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._settings["capture_height"])

        if not cap.isOpened():
            self.error_occurred.emit("Could not open webcam.")
            self._running = False
            return

        # Cached tracker & swipe detector (rebuilt when relevant settings change)
        tracker, swipe_detector = self._build_components()
        prev_settings_sig = self._settings_sig()

        fps_times = deque(maxlen=30)
        prev_time = time.time()

        while self._running:
            # ----- Rebuild components if key settings changed -----
            sig = self._settings_sig()
            if sig != prev_settings_sig:
                tracker, swipe_detector = self._build_components()
                prev_settings_sig = sig

            # ----- Grab frame -----
            ret, frame = cap.read()
            if not ret:
                continue

            # ----- FPS tracking -----
            now = time.time()
            fps_times.append(now - prev_time)
            prev_time = now
            if len(fps_times) >= 10:
                avg_fps = 1.0 / (sum(fps_times) / len(fps_times))
                self.fps_updated.emit(avg_fps)

            frame = cv2.flip(frame, 1)

            # ----- Hand detection -----
            if self._settings["show_landmarks"]:
                frame = tracker.find_hands(frame)
            else:
                # Still detect, just skip drawing the landmarks
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tracker.results = tracker.hands.process(rgb)

            hands = tracker.get_landmarks(frame)

            # ----- Gesture recognition -----
            total_hands = len(hands)

            for hand_index, hand in enumerate(hands):
                gesture = recognize_gesture(hand)
                info, color = GESTURE_INFO.get(gesture, (gesture, (200, 200, 200)))
                wrist_x = hand["landmarks"][0][1]
                wrist_y = hand["landmarks"][0][2]
                label = hand.get("label", "")
                display_text = f"{gesture}: {info}"

                # ----- Emit gesture event for history log -----
                ts = datetime.now().strftime("%H:%M:%S")
                self.gesture_detected.emit(ts, gesture, label)

                # ----- Draw label box (same logic as main.py) -----
                if self._settings["show_gesture_labels"]:
                    self._draw_label_box(
                        frame,
                        total_hands,
                        hand_index,
                        wrist_x,
                        wrist_y,
                        label,
                        display_text,
                        color,
                        gesture,
                    )

            # ----- Swipe detection -----
            if hands:
                swipe = swipe_detector.update(hands[0]["landmarks"])
                if swipe_detector.just_detected:
                    self.swipe_detected.emit(swipe)

            # ----- Swipe overlay (before FPS overlay) -----
            result_frame = draw_swipe_overlay(frame, swipe_detector)
            if result_frame is not None:
                frame = result_frame

            # ----- FPS overlay -----
            if self._settings.get("show_fps", True) and len(fps_times) >= 10:
                fps = 1.0 / (sum(fps_times) / len(fps_times))
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 100),
                    2,
                    cv2.LINE_AA,
                )

            # ----- Emit frame as QImage -----
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.frame_ready.emit(qimg)

        cap.release()
        tracker.release()

    @pyqtSlot()
    def stop(self):
        self._running = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _settings_sig(self):
        """Return a hashable signature of settings that require rebuilding."""
        s = self._settings
        return (s["max_hands"], s["detection_confidence"], s["tracking_confidence"])

    def _build_components(self):
        s = self._settings
        tracker = HandTracker(
            max_hands=s["max_hands"],
            detection_confidence=s["detection_confidence"],
            tracking_confidence=s["tracking_confidence"],
        )
        swipe = SwipeDetector(
            swipe_threshold=s["swipe_threshold"],
            cooldown=s["swipe_cooldown"],
        )
        return tracker, swipe

    @staticmethod
    def _draw_label_box(
        frame, total_hands, hand_index, wrist_x, wrist_y,
        label, display_text, color, gesture,
    ):
        """Draw a smart-positioned label box (mirrors main.py logic)."""
        h, w, _ = frame.shape
        BOX_W, BOX_H = 300, 120
        STATUS_BAR_H = 45
        PAD = 8

        if total_hands == 1:
            bx = wrist_x - BOX_W // 2
            by = wrist_y - BOX_H - 10
        else:
            if label == "Left" or (label == "" and hand_index == 0):
                bx = wrist_x + 30
                by = wrist_y - BOX_H // 2
            else:
                bx = wrist_x - BOX_W - 30
                by = wrist_y - BOX_H // 2

        bx = max(PAD, min(bx, w - BOX_W - PAD))
        if by < STATUS_BAR_H:
            by = wrist_y + 20
        by = max(STATUS_BAR_H, min(by, h - BOX_H - PAD))

        cv2.rectangle(frame, (bx, by), (bx + BOX_W, by + BOX_H), (0, 0, 0), -1)
        cv2.rectangle(frame, (bx, by), (bx + BOX_W, by + BOX_H), color, 2)

        tx = bx + 12
        cv2.putText(frame, display_text, (tx, by + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3, cv2.LINE_AA)
        cv2.putText(frame, f"[{label}]", (tx, by + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Dashboard — main GUI window
# ---------------------------------------------------------------------------


class Dashboard(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Gesture Recognition — Dashboard")
        self.resize(1400, 820)

        # ----- Shared settings (read by worker thread) -----
        self.settings = DEFAULT_PROFILE.copy()
        self._settings_lock_ = True  # prevent update cascades during GUI init

        # ----- UI state -----
        self._history = deque(maxlen=MAX_HISTORY_ROWS)
        self._worker = None
        self._thread = None

        self._build_ui()
        self._settings_lock_ = False

        # ----- Start camera -----
        self._start_camera()

    # ==================================================================
    # UI construction
    # ==================================================================

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(8)

        # --- Left: camera preview ---
        self._camera_label = QLabel()
        self._camera_label.setMinimumSize(640, 480)
        self._camera_label.setAlignment(Qt.AlignCenter)
        self._camera_label.setStyleSheet(
            "background-color: #1e1e1e; border: 1px solid #444;"
        )
        self._camera_label.setText("Starting camera…")

        # --- Right: tabbed panel ---
        right_tabs = QTabWidget()
        right_tabs.setMinimumWidth(360)
        right_tabs.addTab(self._build_settings_tab(), "Settings")
        right_tabs.addTab(self._build_history_tab(), "History")
        right_tabs.addTab(self._build_profiles_tab(), "Profiles")

        splitter = QWidget()
        splitter_layout = QHBoxLayout(splitter)
        splitter_layout.setContentsMargins(0, 0, 0, 0)
        splitter_layout.addWidget(self._camera_label, stretch=3)
        splitter_layout.addWidget(right_tabs, stretch=1)

        main_layout.addWidget(splitter)

        # --- Status bar ---
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status_label = QLabel("Ready")
        self._fps_label = QLabel("FPS: --")
        self._status.addWidget(self._status_label)
        self._status.addPermanentWidget(self._fps_label)

    # ----------------------------------------------------------------
    # Settings tab
    # ----------------------------------------------------------------

    def _build_settings_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        form = QFormLayout(container)
        form.setSpacing(10)
        form.setContentsMargins(12, 12, 12, 12)

        # --- Swipe threshold ---
        self._swipe_thresh_slider = QSlider(Qt.Horizontal)
        self._swipe_thresh_slider.setRange(30, 300)
        self._swipe_thresh_slider.setValue(self.settings["swipe_threshold"])
        self._swipe_thresh_slider.valueChanged.connect(self._on_setting_changed)
        self._swipe_thresh_val = QLabel(str(self.settings["swipe_threshold"]))
        self._swipe_thresh_slider.valueChanged.connect(
            lambda v: self._swipe_thresh_val.setText(str(v))
        )
        row = QHBoxLayout()
        row.addWidget(self._swipe_thresh_slider)
        row.addWidget(self._swipe_thresh_val)
        form.addRow("Swipe threshold (px):", row)

        # --- Swipe cooldown ---
        self._swipe_cd_slider = QSlider(Qt.Horizontal)
        self._swipe_cd_slider.setRange(2, 20)  # 0.2 – 2.0 seconds
        self._swipe_cd_slider.setValue(int(self.settings["swipe_cooldown"] * 10))
        self._swipe_cd_slider.valueChanged.connect(self._on_setting_changed)
        self._swipe_cd_val = QLabel(f"{self.settings['swipe_cooldown']:.1f}s")
        self._swipe_cd_slider.valueChanged.connect(
            lambda v: self._swipe_cd_val.setText(f"{v / 10:.1f}s")
        )
        row = QHBoxLayout()
        row.addWidget(self._swipe_cd_slider)
        row.addWidget(self._swipe_cd_val)
        form.addRow("Swipe cooldown:", row)

        # --- Max hands ---
        self._max_hands_spin = QSpinBox()
        self._max_hands_spin.setRange(1, 4)
        self._max_hands_spin.setValue(self.settings["max_hands"])
        self._max_hands_spin.valueChanged.connect(self._on_setting_changed)
        form.addRow("Max hands:", self._max_hands_spin)

        # --- Detection confidence ---
        self._detect_conf_slider = QSlider(Qt.Horizontal)
        self._detect_conf_slider.setRange(30, 95)
        self._detect_conf_slider.setValue(int(self.settings["detection_confidence"] * 100))
        self._detect_conf_slider.valueChanged.connect(self._on_setting_changed)
        self._detect_conf_val = QLabel(f"{self.settings['detection_confidence']:.2f}")
        self._detect_conf_slider.valueChanged.connect(
            lambda v: self._detect_conf_val.setText(f"{v / 100:.2f}")
        )
        row = QHBoxLayout()
        row.addWidget(self._detect_conf_slider)
        row.addWidget(self._detect_conf_val)
        form.addRow("Detection confidence:", row)

        # --- Tracking confidence ---
        self._track_conf_slider = QSlider(Qt.Horizontal)
        self._track_conf_slider.setRange(30, 95)
        self._track_conf_slider.setValue(int(self.settings["tracking_confidence"] * 100))
        self._track_conf_slider.valueChanged.connect(self._on_setting_changed)
        self._track_conf_val = QLabel(f"{self.settings['tracking_confidence']:.2f}")
        self._track_conf_slider.valueChanged.connect(
            lambda v: self._track_conf_val.setText(f"{v / 100:.2f}")
        )
        row = QHBoxLayout()
        row.addWidget(self._track_conf_slider)
        row.addWidget(self._track_conf_val)
        form.addRow("Tracking confidence:", row)

        # --- Toggles ---
        self._show_landmarks_cb = QCheckBox("Draw hand landmarks on video")
        self._show_landmarks_cb.setChecked(self.settings["show_landmarks"])
        self._show_landmarks_cb.toggled.connect(self._on_setting_changed)
        form.addRow(self._show_landmarks_cb)

        self._show_labels_cb = QCheckBox("Show gesture label boxes")
        self._show_labels_cb.setChecked(self.settings["show_gesture_labels"])
        self._show_labels_cb.toggled.connect(self._on_setting_changed)
        form.addRow(self._show_labels_cb)

        self._show_fps_cb = QCheckBox("Show FPS counter")
        self._show_fps_cb.setChecked(self.settings["show_fps"])
        self._show_fps_cb.toggled.connect(self._on_setting_changed)
        form.addRow(self._show_fps_cb)

        # --- Camera index ---
        self._camera_combo = QComboBox()
        for i in range(4):
            self._camera_combo.addItem(f"Camera {i}")
        self._camera_combo.setCurrentIndex(self.settings["camera_index"])
        self._camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        form.addRow("Camera:", self._camera_combo)

        scroll.setWidget(container)
        return scroll

    # ----------------------------------------------------------------
    # History tab
    # ----------------------------------------------------------------

    def _build_history_tab(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)

        self._history_table = QTableWidget(0, 3)
        self._history_table.setHorizontalHeaderLabels(["Time", "Gesture", "Hand"])
        self._history_table.horizontalHeader().setStretchLastSection(True)
        self._history_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self._history_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self._history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._history_table.setAlternatingRowColors(True)
        self._history_table.verticalHeader().setVisible(False)

        btn_row = QHBoxLayout()
        clear_btn = QPushButton("Clear History")
        clear_btn.clicked.connect(self._clear_history)
        export_btn = QPushButton("Export…")
        export_btn.clicked.connect(self._export_history)
        btn_row.addWidget(clear_btn)
        btn_row.addWidget(export_btn)
        btn_row.addStretch()

        layout.addWidget(self._history_table)
        layout.addLayout(btn_row)

        return container

    # ----------------------------------------------------------------
    # Profiles tab
    # ----------------------------------------------------------------

    def _build_profiles_tab(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # --- Profile list ---
        self._profile_list = QComboBox()
        self._refresh_profile_list()

        save_btn = QPushButton("💾  Save Current Settings…")
        save_btn.clicked.connect(self._save_profile)

        load_btn = QPushButton("📂  Load Profile")
        load_btn.clicked.connect(self._load_profile)

        reset_btn = QPushButton("↺  Reset to Defaults")
        reset_btn.clicked.connect(self._reset_profile)

        delete_btn = QPushButton("🗑  Delete Selected Profile")
        delete_btn.clicked.connect(self._delete_profile)

        layout.addWidget(QLabel("Saved profiles:"))
        layout.addWidget(self._profile_list)
        layout.addWidget(save_btn)
        layout.addWidget(load_btn)
        layout.addSpacing(16)
        layout.addWidget(reset_btn)
        layout.addWidget(delete_btn)
        layout.addStretch()

        return container

    # ==================================================================
    # Camera lifecycle
    # ==================================================================

    def _start_camera(self):
        self._stop_camera()
        self._thread = QThread()
        self._worker = CameraWorker(self.settings)
        self._worker.moveToThread(self._thread)

        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.gesture_detected.connect(self._on_gesture_detected)
        self._worker.swipe_detected.connect(self._on_swipe_detected)
        self._worker.fps_updated.connect(self._on_fps_updated)
        self._worker.error_occurred.connect(self._on_camera_error)
        self._thread.started.connect(self._worker.start)

        self._thread.start()

    def _stop_camera(self):
        if self._worker is not None:
            self._worker.stop()
            self._worker = None
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
            self._thread = None

    # ==================================================================
    # Slots
    # ==================================================================

    def _on_frame_ready(self, qimg: QImage):
        pix = QPixmap.fromImage(qimg)
        # Scale to fit the label while preserving aspect ratio
        scaled = pix.scaled(
            self._camera_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._camera_label.setPixmap(scaled)

    def _on_gesture_detected(self, ts: str, gesture: str, hand_label: str):
        # Deduplicate: only log when the gesture changes for this hand
        last_key = f"gesture_{hand_label}"
        last = getattr(self, last_key, None)
        if last == gesture:
            return
        setattr(self, last_key, gesture)

        info = GESTURE_INFO.get(gesture, (gesture, (0, 0, 0)))[0]
        display = f"{gesture}: {info}"
        self._history.append((ts, display, hand_label or "—"))
        self._update_history_table()

    def _on_swipe_detected(self, direction: str):
        self._status_label.setText(f"Swipe {direction} detected!")

    def _on_fps_updated(self, fps: float):
        self._fps_label.setText(f"FPS: {fps:.1f}")

    def _on_camera_error(self, msg: str):
        self._camera_label.setText(msg)
        QMessageBox.critical(self, "Camera Error", msg)

    # ----------------------------------------------------------------
    # Settings changes
    # ----------------------------------------------------------------

    def _on_setting_changed(self):
        if self._settings_lock_:
            return
        self.settings["swipe_threshold"] = self._swipe_thresh_slider.value()
        self.settings["swipe_cooldown"] = self._swipe_cd_slider.value() / 10
        self.settings["max_hands"] = self._max_hands_spin.value()
        self.settings["detection_confidence"] = self._detect_conf_slider.value() / 100
        self.settings["tracking_confidence"] = self._track_conf_slider.value() / 100
        self.settings["show_landmarks"] = self._show_landmarks_cb.isChecked()
        self.settings["show_gesture_labels"] = self._show_labels_cb.isChecked()
        self.settings["show_fps"] = self._show_fps_cb.isChecked()
        # Worker picks up max_hands/confidence changes on next frame via _settings_sig

    def _on_camera_changed(self, idx: int):
        self.settings["camera_index"] = idx
        self._status_label.setText("Restarting camera…")
        self._start_camera()

    # ----------------------------------------------------------------
    # History
    # ----------------------------------------------------------------

    def _update_history_table(self):
        self._history_table.setRowCount(len(self._history))
        for row, (ts, gesture, hand) in enumerate(self._history):
            self._history_table.setItem(row, 0, QTableWidgetItem(ts))
            self._history_table.setItem(row, 1, QTableWidgetItem(gesture))
            self._history_table.setItem(row, 2, QTableWidgetItem(hand))

    def _clear_history(self):
        self._history.clear()
        self._history_table.setRowCount(0)

    def _export_history(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export History", "gesture_history.csv", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "w") as f:
                f.write("Time,Gesture,Hand\n")
                for ts, gesture, hand in self._history:
                    f.write(f"{ts},{gesture},{hand}\n")
            self._status_label.setText(f"History exported to {path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    # ----------------------------------------------------------------
    # Profiles
    # ----------------------------------------------------------------

    def _refresh_profile_list(self):
        self._profile_list.clear()
        try:
            files = sorted(f for f in os.listdir(PROFILES_DIR) if f.endswith(".json"))
        except FileNotFoundError:
            files = []
        self._profile_list.addItems(files)
        self._profile_list.addItem("— Default —")

    def _save_profile(self):
        name, ok = QFileDialog.getSaveFileName(
            self, "Save Profile",
            os.path.join(PROFILES_DIR, "my_profile.json"),
            "JSON (*.json)",
        )
        if not ok or not name:
            return
        try:
            with open(name, "w") as f:
                json.dump(self.settings, f, indent=2)
            self._status_label.setText(f"Profile saved: {os.path.basename(name)}")
            self._refresh_profile_list()
        except Exception as e:
            QMessageBox.warning(self, "Save Error", str(e))

    def _load_profile(self):
        name = self._profile_list.currentText()
        if not name or name == "— Default —":
            return
        path = os.path.join(PROFILES_DIR, name)
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self.settings.update(data)
            self._sync_ui_from_settings()
            # Restart camera so new settings take full effect
            self._start_camera()
            self._status_label.setText(f"Profile loaded: {name}")
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))

    def _reset_profile(self):
        reply = QMessageBox.question(
            self, "Reset to Defaults",
            "Reset all settings to their default values?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.settings = DEFAULT_PROFILE.copy()
        self._sync_ui_from_settings()
        self._start_camera()
        self._clear_history()
        self._status_label.setText("Reset to default settings")

    def _delete_profile(self):
        name = self._profile_list.currentText()
        if not name or name == "— Default —":
            return
        path = os.path.join(PROFILES_DIR, name)
        reply = QMessageBox.question(
            self, "Delete Profile",
            f"Delete '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            os.remove(path)
            self._refresh_profile_list()
            self._status_label.setText(f"Deleted: {name}")
        except Exception as e:
            QMessageBox.warning(self, "Delete Error", str(e))

    def _sync_ui_from_settings(self):
        """Push settings dict back into all UI widgets."""
        self._settings_lock_ = True
        s = self.settings
        self._swipe_thresh_slider.setValue(s["swipe_threshold"])
        self._swipe_cd_slider.setValue(int(s["swipe_cooldown"] * 10))
        self._max_hands_spin.setValue(s["max_hands"])
        self._detect_conf_slider.setValue(int(s["detection_confidence"] * 100))
        self._track_conf_slider.setValue(int(s["tracking_confidence"] * 100))
        self._show_landmarks_cb.setChecked(s["show_landmarks"])
        self._show_labels_cb.setChecked(s["show_gesture_labels"])
        self._show_fps_cb.setChecked(s["show_fps"])
        self._camera_combo.setCurrentIndex(s.get("camera_index", 0))
        self._swipe_thresh_val.setText(str(s["swipe_threshold"]))
        self._swipe_cd_val.setText(f"{s['swipe_cooldown']:.1f}s")
        self._detect_conf_val.setText(f"{s['detection_confidence']:.2f}")
        self._track_conf_val.setText(f"{s['tracking_confidence']:.2f}")
        self._settings_lock_ = False

    # ==================================================================
    # Window close
    # ==================================================================

    def closeEvent(self, event: QCloseEvent):
        self._stop_camera()
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication([])
    app.setStyle("Fusion")
    win = Dashboard()
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
