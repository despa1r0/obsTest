from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections


@dataclass(slots=True)
class GestureResult:
    gesture_name: str
    handedness: str


class HandGestureDetector:
    """Detects a hand and classifies a few simple gestures."""

    def __init__(
        self,
        model_path: str | Path,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        resolved_model_path = Path(model_path).resolve()
        if not resolved_model_path.exists():
            raise FileNotFoundError(
                f"Hand landmarker model was not found: {resolved_model_path}"
            )

        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(resolved_model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._hands = vision.HandLandmarker.create_from_options(options)

    def close(self) -> None:
        self._hands.close()

    def process_frame(self, frame, timestamp_ms: int):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self._hands.detect_for_video(mp_image, timestamp_ms)

    def draw_landmarks(self, frame, results) -> None:
        if not results.hand_landmarks:
            return

        for hand_landmarks in results.hand_landmarks:
            drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                HandLandmarksConnections.HAND_CONNECTIONS,
            )

    def classify(self, results) -> list[GestureResult]:
        classified: list[GestureResult] = []
        if not results.hand_landmarks or not results.handedness:
            return classified

        for hand_landmarks, handedness in zip(
            results.hand_landmarks,
            results.handedness,
        ):
            label = handedness[0].category_name
            gesture_name = self._classify_from_landmarks(hand_landmarks, label)
            classified.append(GestureResult(gesture_name=gesture_name, handedness=label))

        return classified

    def _classify_from_landmarks(self, landmarks: Iterable, handedness: str) -> str:
        points = list(landmarks)

        thumb_open = self._is_thumb_extended(points, handedness)
        index_open = self._is_finger_extended(
            points,
            tip_id=HandLandmark.INDEX_FINGER_TIP,
            pip_id=HandLandmark.INDEX_FINGER_PIP,
        )
        middle_open = self._is_finger_extended(
            points,
            tip_id=HandLandmark.MIDDLE_FINGER_TIP,
            pip_id=HandLandmark.MIDDLE_FINGER_PIP,
        )
        ring_open = self._is_finger_extended(
            points,
            tip_id=HandLandmark.RING_FINGER_TIP,
            pip_id=HandLandmark.RING_FINGER_PIP,
        )
        pinky_open = self._is_finger_extended(
            points,
            tip_id=HandLandmark.PINKY_TIP,
            pip_id=HandLandmark.PINKY_PIP,
        )

        opened = [thumb_open, index_open, middle_open, ring_open, pinky_open]

        if all(opened):
            return "open_palm"
        if not any(opened):
            return "fist"
        if index_open and middle_open and not ring_open and not pinky_open and not thumb_open:
            return "peace"
        if thumb_open and not index_open and not middle_open and not ring_open and not pinky_open:
            return "thumbs_up"
        return "unknown"

    @staticmethod
    def _is_finger_extended(points, tip_id, pip_id) -> bool:
        return points[tip_id].y < points[pip_id].y

    @staticmethod
    def _is_thumb_extended(points, handedness: str) -> bool:
        thumb_tip = points[HandLandmark.THUMB_TIP]
        thumb_ip = points[HandLandmark.THUMB_IP]
        wrist = points[HandLandmark.WRIST]

        horizontal_extension = (
            thumb_tip.x > thumb_ip.x if handedness == "Right" else thumb_tip.x < thumb_ip.x
        )
        vertical_extension = thumb_tip.y < wrist.y
        return horizontal_extension or vertical_extension
