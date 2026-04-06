from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections


@dataclass(slots=True)
class EmotionResult:
    emotion_name: str
    confidence_hint: float


class FaceEmotionDetector:
    """Classifies five basic emotions from MediaPipe face blendshapes."""

    def __init__(
        self,
        model_path: str | Path,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
    ) -> None:
        resolved_model_path = Path(model_path).resolve()
        if not resolved_model_path.exists():
            raise FileNotFoundError(
                f"Face landmarker model was not found: {resolved_model_path}"
            )

        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(resolved_model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            min_face_presence_confidence=min_presence_confidence,
            output_face_blendshapes=True,
        )
        self._faces = vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        self._faces.close()

    def process_frame(self, frame, timestamp_ms: int):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self._faces.detect_for_video(mp_image, timestamp_ms)

    def draw_landmarks(self, frame, results) -> None:
        if not results.face_landmarks:
            return

        for face_landmarks in results.face_landmarks:
            drawing_utils.draw_landmarks(
                frame,
                face_landmarks,
                FaceLandmarksConnections.FACE_LANDMARKS_LIPS
                + FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE
                + FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE
                + FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW
                + FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW
                + FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_utils.DrawingSpec(
                    color=(255, 180, 0),
                    thickness=1,
                    circle_radius=1,
                ),
                is_drawing_landmarks=False,
            )

    def classify(self, results) -> EmotionResult | None:
        if not results.face_blendshapes:
            return None

        scores = {
            self._normalize_name(category.category_name): category.score
            for category in results.face_blendshapes[0]
        }
        emotion_scores = {
            "happy": self._average(
                scores,
                "mouthsmileleft",
                "mouthsmileright",
            ),
            "sad": self._average(
                scores,
                "mouthfrownleft",
                "mouthfrownright",
            )
            + scores.get("browinnerup", 0.0) * 0.25,
            "angry": self._average(
                scores,
                "browdownleft",
                "browdownright",
            )
            + self._average(scores, "nosesneerleft", "nosesneerright") * 0.35
            + scores.get("mouthpressleft", 0.0) * 0.15
            + scores.get("mouthpressright", 0.0) * 0.15,
            "surprised": scores.get("jawopen", 0.0)
            + self._average(scores, "eyewideleft", "eyewideright") * 0.35
            + scores.get("browinnerup", 0.0) * 0.25,
            "neutral": scores.get("neutral", 0.0),
        }

        best_emotion = max(emotion_scores, key=emotion_scores.get)
        best_score = emotion_scores[best_emotion]
        if best_emotion != "neutral" and best_score < 0.25:
            return EmotionResult(emotion_name="neutral", confidence_hint=1 - best_score)
        return EmotionResult(emotion_name=best_emotion, confidence_hint=best_score)

    @staticmethod
    def _normalize_name(name: str | None) -> str:
        return (name or "").replace("_", "").replace("-", "").lower()

    @staticmethod
    def _average(scores: dict[str, float], *keys: str) -> float:
        values = [scores.get(key, 0.0) for key in keys]
        return sum(values) / len(values)
