from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2

from app.gesture_detector import HandGestureDetector


WINDOW_NAME = "Gesture Detection MVP"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
QUIT_KEYS = {27, ord("q"), ord("Q")}


def draw_labels(frame, gestures) -> None:
    cv2.putText(
        frame,
        "Press Esc to stop",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if not gestures:
        cv2.putText(
            frame,
            "No hand detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return

    for index, gesture in enumerate(gestures):
        text = f"{gesture.handedness}: {gesture.gesture_name}"
        y = 40 + index * 35
        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    capture = cv2.VideoCapture(0)
    detector = HandGestureDetector(model_path=MODEL_PATH)

    if not capture.isOpened():
        raise RuntimeError("Could not open the default camera.")

    start_time = perf_counter()

    try:
        while True:
            success, frame = capture.read()
            if not success:
                raise RuntimeError("Could not read a frame from the camera.")

            frame = cv2.flip(frame, 1)
            timestamp_ms = int((perf_counter() - start_time) * 1000)
            results = detector.process_frame(frame, timestamp_ms)
            detector.draw_landmarks(frame, results)
            gestures = detector.classify(results)
            draw_labels(frame, gestures)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in QUIT_KEYS:
                break
    except KeyboardInterrupt:
        pass
    finally:
        detector.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
