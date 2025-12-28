import cv2
import time
import mediapipe as mp
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# ==== CONFIG ====
MODEL_PATH = str((Path(__file__).resolve().parent / "hand_landmarker.task"))
CAMERA_INDEX = 0


# ==== MEDIAPIPE TASKS TYPE ALIASES ====
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# Hand skeleton connections
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
]


@dataclass
class HandDrawingConfig:
    landmark_radius: int = 3
    landmark_thickness: int = -1
    connection_thickness: int = 2


def draw_hand_landmarks(
    frame,
    hand_landmarks,
    handedness_label: str = "",
    cfg: HandDrawingConfig = HandDrawingConfig(),
):
    """Draw landmarks + skeleton for a single hand on the BGR frame."""
    h, w, _ = frame.shape

    # Convert normalized coordinates to pixel coords
    pts = []
    for lm in hand_landmarks:
        x_px = int(lm.x * w)
        y_px = int(lm.y * h)
        pts.append((x_px, y_px))
        # Draw landmark point
        cv2.circle(
            frame,
            (x_px, y_px),
            cfg.landmark_radius,
            (0, 255, 0),
            cfg.landmark_thickness,
        )

    # Draw connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        x1, y1 = pts[start_idx]
        x2, y2 = pts[end_idx]
        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            cfg.connection_thickness,
        )

    # Optional handedness label near wrist (landmark 0)
    if handedness_label:
        x0, y0 = pts[0]
        cv2.putText(
            frame,
            handedness_label,
            (x0 - 20, y0 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )


def hand_tracker():
    # Open webcam
    cam = cv2.VideoCapture(CAMERA_INDEX)
    if not cam.isOpened():
        print(f"Error: cannot open camera index {CAMERA_INDEX}")
        return

    # Create HandLandmarker in VIDEO mode
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        start_time = time.time()

        while True:
            success, frame = cam.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Convert BGR (OpenCV) -> RGB (MediaPipe expects SRGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Wrap as MediaPipe Image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            # Timestamp in ms
            timestamp_ms = int((time.time() - start_time) * 1000)

            # Run the hand landmarker
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Draw all detected hands on the original BGR frame
            if result.hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
                    # Get handedness label (Left/Right) if available
                    label = ""
                    if result.handedness and len(result.handedness) > hand_idx:
                        # handedness[hand_idx] is a list of Category objects
                        top_category = result.handedness[hand_idx][0]
                        # Attributes are usually .category_name and .score
                        label = f"{top_category.category_name} ({top_category.score:.2f})"

                    draw_hand_landmarks(frame, hand_landmarks, label)

            # Flip for selfie view
            display_frame = cv2.flip(frame, 1)
            cv2.imshow("Hand Tracker", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cam.release()
    cv2.destroyAllWindows()


def main():
    hand_tracker()


if __name__ == "__main__":
    main()
