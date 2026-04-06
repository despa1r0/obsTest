from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2

from app.face_emotion_detector import EmotionResult, FaceEmotionDetector
from app.gesture_detector import GestureResult, HandGestureDetector
from app.meme_library import MemeLibrary, MemeMatch


WINDOW_NAME = "Gesture + Face Meme MVP"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
FACE_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
MEMES_DIR = Path(__file__).resolve().parent.parent / "memes"
QUIT_KEYS = {27, ord("q"), ord("Q")}


def build_tags(gestures: list[GestureResult], emotion: EmotionResult | None) -> list[str]:
    tags = [gesture.gesture_name for gesture in gestures if gesture.gesture_name != "unknown"]
    tags.extend(
        f"{gesture.handedness.lower()}_{gesture.gesture_name}"
        for gesture in gestures
        if gesture.gesture_name != "unknown"
    )
    if emotion is not None:
        tags.append(emotion.emotion_name)
    return sorted(set(tags))


def draw_labels(
    frame,
    gestures: list[GestureResult],
    emotion: EmotionResult | None,
    meme_match: MemeMatch | None,
    tags: list[str],
) -> None:
    hud_lines: list[tuple[str, tuple[int, int, int]]] = [
        ("Press Esc to stop", (255, 255, 255)),
    ]

    if gestures:
        for gesture in gestures:
            hud_lines.append(
                (f"Hand {gesture.handedness}: {gesture.gesture_name}", (0, 255, 0))
            )
    else:
        hud_lines.append(("Hands: not detected", (0, 0, 255)))

    if emotion is None:
        hud_lines.append(("Face emotion: not detected", (0, 180, 255)))
    else:
        score = f"{emotion.confidence_hint:.2f}"
        hud_lines.append((f"Face emotion: {emotion.emotion_name} ({score})", (255, 220, 0)))

    if meme_match is None:
        hud_lines.append(("Meme match: no tagged image yet", (180, 180, 180)))
    else:
        hud_lines.append((f"Meme match: {meme_match.file_name}", (255, 255, 0)))
        hud_lines.append((f"Matched tags: {', '.join(meme_match.matched_tags)}", (255, 255, 0)))

    if tags:
        hud_lines.append((f"Frame tags: {', '.join(tags)}", (210, 210, 210)))

    for index, (text, color) in enumerate(hud_lines):
        y = 35 + index * 28
        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )


def should_stop(window_name: str, key: int) -> bool:
    if key in QUIT_KEYS:
        return True
    return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1


def main() -> None:
    capture = cv2.VideoCapture(0)
    hand_detector = HandGestureDetector(model_path=MODEL_PATH, max_num_hands=2)
    face_detector = FaceEmotionDetector(model_path=FACE_MODEL_PATH, max_num_faces=1)
    meme_library = MemeLibrary(MEMES_DIR)

    if not capture.isOpened():
        raise RuntimeError("Could not open the default camera.")

    print("Camera stream started.")
    print("Hands: up to 2 hands. Face: 5 basic emotions.")
    print("Press Esc to stop the script. You can also press Q or close the window.")

    if not meme_library.has_entries():
        print("No meme tags found yet. Add memes/tags.json to enable matching.")

    start_time = perf_counter()

    try:
        while True:
            success, frame = capture.read()
            if not success:
                raise RuntimeError("Could not read a frame from the camera.")

            frame = cv2.flip(frame, 1)
            timestamp_ms = int((perf_counter() - start_time) * 1000)

            hand_results = hand_detector.process_frame(frame, timestamp_ms)
            face_results = face_detector.process_frame(frame, timestamp_ms)

            hand_detector.draw_landmarks(frame, hand_results)
            face_detector.draw_landmarks(frame, face_results)

            gestures = hand_detector.classify(hand_results)
            emotion = face_detector.classify(face_results)
            tags = build_tags(gestures, emotion)
            meme_match = meme_library.find_best_match(tags)

            draw_labels(frame, gestures, emotion, meme_match, tags)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if should_stop(WINDOW_NAME, key):
                break
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping camera stream.")
        face_detector.close()
        hand_detector.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
