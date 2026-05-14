from __future__ import annotations

import os
from pathlib import Path


def evaluate_video(
    weights: Path,
    video_path: Path,
    conf: float,
    show: bool,
    save: bool,
    output_path: Path,
) -> Path | None:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / ".ultralytics"))

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing video inference package. Activate the local environment and run "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    if not weights.exists():
        raise FileNotFoundError(f"Model weights not found: {weights}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not show and not save:
        raise ValueError("Nothing to do: enable playback or pass --save.")

    model = YOLO(str(weights))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not create output video: {output_path}")

    window_name = "YOLO object dot evaluation"
    if show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = model.predict(frame, conf=conf, verbose=False)[0]
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                best_index = None
                best_score = -1.0
                for index, (class_id, score) in enumerate(zip(boxes.cls.tolist(), boxes.conf.tolist())):
                    if int(class_id) == 0 and float(score) > best_score:
                        best_index = index
                        best_score = float(score)

                if best_index is not None:
                    x1, y1, x2, y2 = boxes.xyxy[best_index].tolist()
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    cv2.circle(frame, (center_x, center_y), 10, (0, 0, 255), thickness=-1)
                    cv2.circle(frame, (center_x, center_y), 16, (255, 255, 255), thickness=2)
                    cv2.putText(
                        frame,
                        f"object {best_score:.2f}",
                        (center_x + 14, max(24, center_y - 14)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

            if writer is not None:
                writer.write(frame)

            if show:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    if save:
        print(f"Annotated video saved: {output_path}")
        return output_path

    return None
