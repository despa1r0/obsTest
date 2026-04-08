from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from app.face_emotion_detector import EmotionResult, FaceEmotionDetector
from app.gesture_detector import GestureResult, HandGestureDetector
from app.meme_library import MemeLibrary, MemeMatch


WINDOW_NAME = "Gesture + Face Meme MVP"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
FACE_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
MEMES_DIR = Path(__file__).resolve().parent.parent / "memes"
QUIT_KEYS = {27, ord("q"), ord("Q")}
PREVIEW_PANEL_WIDTH = 360
ACTIVATION_DELAY_SECONDS = 0.75
FADE_DURATION_SECONDS = 0.8
SPECIAL_REVEAL_DURATION_SECONDS = 5.1
SPECIAL_REVEAL_FILES = {"cinema.png"}


@dataclass(slots=True)
class DisplayState:
    active_match: MemeMatch | None = None
    candidate_match: MemeMatch | None = None
    candidate_since: float = 0.0
    previous_match: MemeMatch | None = None
    transition_started_at: float = 0.0
    active_since: float = 0.0


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


def same_match(left: MemeMatch | None, right: MemeMatch | None) -> bool:
    if left is None or right is None:
        return left is right
    return left.file_name == right.file_name


def smoothstep(progress: float) -> float:
    clipped = max(0.0, min(1.0, progress))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def is_special_reveal_match(match: MemeMatch | None) -> bool:
    return match is not None and match.file_name.lower() in SPECIAL_REVEAL_FILES


def update_display_state(
    state: DisplayState,
    raw_match: MemeMatch | None,
    now_seconds: float,
) -> None:
    if same_match(raw_match, state.active_match):
        state.candidate_match = raw_match
        state.candidate_since = now_seconds
        return

    if not same_match(raw_match, state.candidate_match):
        state.candidate_match = raw_match
        state.candidate_since = now_seconds
        return

    if now_seconds - state.candidate_since < ACTIVATION_DELAY_SECONDS:
        return

    state.previous_match = state.active_match
    state.active_match = state.candidate_match
    state.transition_started_at = now_seconds
    state.active_since = now_seconds


def get_match_image(
    match: MemeMatch | None,
    image_cache: dict[str, np.ndarray | None],
) -> np.ndarray | None:
    if match is None:
        return None
    if match.file_name not in image_cache:
        image_cache[match.file_name] = cv2.imread(str(match.file_path))
    return image_cache[match.file_name]


def fit_image(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return image

    scale = min(max_width / width, max_height / height)
    resized = cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.full((max_height, max_width, 3), 28, dtype=np.uint8)
    y_offset = (max_height - resized.shape[0]) // 2
    x_offset = (max_width - resized.shape[1]) // 2
    canvas[
        y_offset:y_offset + resized.shape[0],
        x_offset:x_offset + resized.shape[1],
    ] = resized
    return canvas


def build_placeholder(preview_width: int, preview_height: int) -> np.ndarray:
    placeholder = np.full((preview_height, preview_width, 3), 36, dtype=np.uint8)
    cv2.putText(
        placeholder,
        "No active meme yet",
        (28, preview_height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )
    return placeholder


def apply_special_reveal(
    image: np.ndarray,
    state: DisplayState,
    now_seconds: float,
) -> np.ndarray:
    if not is_special_reveal_match(state.active_match):
        return image

    reveal_progress = smoothstep(
        (now_seconds - state.active_since) / SPECIAL_REVEAL_DURATION_SECONDS
    )
    background = np.full_like(image, 28)
    return cv2.addWeighted(background, 1 - reveal_progress, image, reveal_progress, 0)


def build_preview_panel(
    frame_height: int,
    state: DisplayState,
    image_cache: dict[str, np.ndarray | None],
    now_seconds: float,
) -> np.ndarray:
    panel = np.full((frame_height, PREVIEW_PANEL_WIDTH, 3), 24, dtype=np.uint8)

    cv2.putText(
        panel,
        "Selected Meme",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    preview_top = 58
    preview_height = frame_height - 170
    preview_width = PREVIEW_PANEL_WIDTH - 36
    placeholder = build_placeholder(preview_width, preview_height)

    active_source = get_match_image(state.active_match, image_cache)
    active_image = fit_image(active_source, preview_width, preview_height) if active_source is not None else placeholder
    active_image = apply_special_reveal(active_image, state, now_seconds)

    transition_elapsed = now_seconds - state.transition_started_at
    if state.active_match is not None and transition_elapsed < FADE_DURATION_SECONDS:
        previous_source = get_match_image(state.previous_match, image_cache)
        previous_image = (
            fit_image(previous_source, preview_width, preview_height)
            if previous_source is not None
            else placeholder
        )
        alpha = smoothstep(transition_elapsed / FADE_DURATION_SECONDS)
        active_image = cv2.addWeighted(previous_image, 1 - alpha, active_image, alpha, 0)

    panel[preview_top:preview_top + preview_height, 18:18 + preview_width] = active_image

    active_name = state.active_match.file_name if state.active_match is not None else "waiting..."
    cv2.putText(
        panel,
        f"Live: {active_name}",
        (18, frame_height - 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    if state.candidate_match is not None and not same_match(state.candidate_match, state.active_match):
        remaining = max(0.0, ACTIVATION_DELAY_SECONDS - (now_seconds - state.candidate_since))
        cv2.putText(
            panel,
            f"Pending: {state.candidate_match.file_name}",
            (18, frame_height - 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (160, 220, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"Hold steady: {remaining:.2f}s",
            (18, frame_height - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (160, 220, 255),
            2,
            cv2.LINE_AA,
        )

    return panel


def draw_labels(
    frame,
    gestures: list[GestureResult],
    emotion: EmotionResult | None,
    raw_match: MemeMatch | None,
    active_match: MemeMatch | None,
    tags: list[str],
) -> None:
    hud_lines: list[tuple[str, tuple[int, int, int]]] = [("Press Esc to stop", (255, 255, 255))]

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

    raw_name = raw_match.file_name if raw_match is not None else "no raw match"
    active_name = active_match.file_name if active_match is not None else "no active meme"
    hud_lines.append((f"Raw match: {raw_name}", (180, 180, 180)))
    hud_lines.append((f"Displayed meme: {active_name}", (255, 255, 0)))

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


def compose_output_frame(camera_frame: np.ndarray, preview_panel: np.ndarray) -> np.ndarray:
    return np.hstack((camera_frame, preview_panel))


def should_stop(window_name: str, key: int) -> bool:
    if key in QUIT_KEYS:
        return True
    return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1


def main() -> None:
    capture = cv2.VideoCapture(0)
    hand_detector = HandGestureDetector(model_path=MODEL_PATH, max_num_hands=2)
    face_detector = FaceEmotionDetector(model_path=FACE_MODEL_PATH, max_num_faces=1)
    meme_library = MemeLibrary(MEMES_DIR)
    display_state = DisplayState()
    image_cache: dict[str, np.ndarray | None] = {}

    if not capture.isOpened():
        raise RuntimeError("Could not open the default camera.")

    print("Camera stream started.")
    print("Hands: up to 2 hands. Face: 5 basic emotions.")
    print("Meme preview is shown on the right with hold-delay smoothing.")
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
            now_seconds = perf_counter() - start_time
            timestamp_ms = int(now_seconds * 1000)

            hand_results = hand_detector.process_frame(frame, timestamp_ms)
            face_results = face_detector.process_frame(frame, timestamp_ms)

            hand_detector.draw_landmarks(frame, hand_results)
            face_detector.draw_landmarks(frame, face_results)

            gestures = hand_detector.classify(hand_results)
            emotion = face_detector.classify(face_results)
            tags = build_tags(gestures, emotion)
            raw_match = meme_library.find_best_match(tags)

            update_display_state(display_state, raw_match, now_seconds)
            draw_labels(frame, gestures, emotion, raw_match, display_state.active_match, tags)

            preview_panel = build_preview_panel(
                frame_height=frame.shape[0],
                state=display_state,
                image_cache=image_cache,
                now_seconds=now_seconds,
            )
            output_frame = compose_output_frame(frame, preview_panel)

            cv2.imshow(WINDOW_NAME, output_frame)
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
