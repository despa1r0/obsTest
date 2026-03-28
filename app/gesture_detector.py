from __future__ import annotations

from dataclasses import dataclass
import math
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


@dataclass(slots=True)
class FingerState:
    thumb_up: bool
    index_extended: bool
    middle_extended: bool
    ring_extended: bool
    pinky_extended: bool

    @property
    def pattern(self) -> tuple[bool, bool, bool, bool]:
        return (
            self.index_extended,
            self.middle_extended,
            self.ring_extended,
            self.pinky_extended,
        )


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
            gesture_name = self._classify_from_landmarks(hand_landmarks)
            classified.append(GestureResult(gesture_name=gesture_name, handedness=label))

        return classified

    def _classify_from_landmarks(self, landmarks: Iterable) -> str:
        points = list(landmarks)
        palm_size = self._get_palm_size(points)
        fingers = FingerState(
            thumb_up=self._is_thumb_up(points, palm_size),
            index_extended=self._is_finger_extended(
                points,
                tip_id=HandLandmark.INDEX_FINGER_TIP,
                pip_id=HandLandmark.INDEX_FINGER_PIP,
                mcp_id=HandLandmark.INDEX_FINGER_MCP,
                palm_size=palm_size,
            ),
            middle_extended=self._is_finger_extended(
                points,
                tip_id=HandLandmark.MIDDLE_FINGER_TIP,
                pip_id=HandLandmark.MIDDLE_FINGER_PIP,
                mcp_id=HandLandmark.MIDDLE_FINGER_MCP,
                palm_size=palm_size,
            ),
            ring_extended=self._is_finger_extended(
                points,
                tip_id=HandLandmark.RING_FINGER_TIP,
                pip_id=HandLandmark.RING_FINGER_PIP,
                mcp_id=HandLandmark.RING_FINGER_MCP,
                palm_size=palm_size,
            ),
            pinky_extended=self._is_finger_extended(
                points,
                tip_id=HandLandmark.PINKY_TIP,
                pip_id=HandLandmark.PINKY_PIP,
                mcp_id=HandLandmark.PINKY_MCP,
                palm_size=palm_size,
            ),
        )

        gesture_by_pattern = {
            (True, True, True, True): "open_palm",
            (True, True, False, False): "peace",
        }

        if fingers.pattern in gesture_by_pattern:
            return gesture_by_pattern[fingers.pattern]
        if fingers.pattern == (False, False, False, False):
            return "thumbs_up" if fingers.thumb_up else "fist"
        return "unknown"

    @staticmethod
    def _get_palm_size(points) -> float:
        wrist = points[HandLandmark.WRIST]
        middle_mcp = points[HandLandmark.MIDDLE_FINGER_MCP]
        return max(
            math.hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y),
            1e-6,
        )

    @staticmethod
    def _is_finger_extended(points, tip_id, pip_id, mcp_id, palm_size: float) -> bool:
        tip = points[tip_id]
        pip = points[pip_id]
        mcp = points[mcp_id]

        tip_above_pip = tip.y < pip.y
        pip_above_mcp = pip.y < mcp.y
        finger_span = mcp.y - tip.y
        return tip_above_pip and pip_above_mcp and finger_span > palm_size * 0.25

    @staticmethod
    def _is_thumb_up(points, palm_size: float) -> bool:
        thumb_tip = points[HandLandmark.THUMB_TIP]
        thumb_ip = points[HandLandmark.THUMB_IP]
        thumb_mcp = points[HandLandmark.THUMB_MCP]
        index_mcp = points[HandLandmark.INDEX_FINGER_MCP]
        wrist = points[HandLandmark.WRIST]

        vertical_order_is_up = thumb_tip.y < thumb_ip.y < thumb_mcp.y
        thumb_is_high_enough = thumb_tip.y < wrist.y - palm_size * 0.05
        thumb_is_centered = abs(thumb_tip.x - thumb_mcp.x) < palm_size * 1.2
        thumb_is_far_from_index_base = abs(thumb_tip.x - index_mcp.x) > palm_size * 0.15
        return (
            vertical_order_is_up
            and thumb_is_high_enough
            and thumb_is_centered
            and thumb_is_far_from_index_base
        )
