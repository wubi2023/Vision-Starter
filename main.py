from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset import validate_dataset, write_dataset_yaml
from src.train import train_model
from src.video_eval import evaluate_video


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a YOLO detector and evaluate colortest.mp4 with a center-dot overlay."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "setup-check",
        help="Validate Python packages, dataset files, and project config.",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train a YOLO detector using all images and labels.",
    )
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--imgsz", type=int, default=640)
    train_parser.add_argument("--batch", type=int, default=4)
    train_parser.add_argument("--model", default="yolov8n.pt")
    train_parser.add_argument("--project", default=str(PROJECT_ROOT / "runs" / "detect"))
    train_parser.add_argument("--name", default="object_train")

    eval_parser = subparsers.add_parser(
        "eval-video",
        help="Play colortest.mp4 with a dot on the detected object center.",
    )
    eval_parser.add_argument(
        "--weights",
        default=str(PROJECT_ROOT / "runs" / "detect" / "object_train" / "weights" / "best.pt"),
        help="Path to trained YOLO weights.",
    )
    eval_parser.add_argument("--video", default=str(PROJECT_ROOT / "colortest.mp4"))
    eval_parser.add_argument("--conf", type=float, default=0.5)
    eval_parser.add_argument("--save", action="store_true")
    eval_parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "runs" / "video_eval" / "colortest_annotated.mp4"),
    )
    eval_parser.add_argument("--no-window", action="store_true", help="Do not open a playback window.")

    all_parser = subparsers.add_parser(
        "all",
        help="Validate, train, then evaluate the video.",
    )
    all_parser.add_argument("--epochs", type=int, default=100)
    all_parser.add_argument("--imgsz", type=int, default=640)
    all_parser.add_argument("--batch", type=int, default=4)
    all_parser.add_argument("--model", default="yolov8n.pt")
    all_parser.add_argument("--conf", type=float, default=0.5)
    all_parser.add_argument("--save", action="store_true")
    all_parser.add_argument("--no-window", action="store_true")

    return parser


def setup_check() -> None:
    report = validate_dataset(PROJECT_ROOT)
    dataset_yaml = write_dataset_yaml(PROJECT_ROOT)

    print(f"Images: {report.image_count}")
    print(f"Labels: {report.label_count}")
    print(f"Objects: {report.object_count}")
    print(f"Dataset config: {dataset_yaml}")

    try:
        import ultralytics  # noqa: F401
        import cv2  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing package: "
            f"{exc.name}. Activate the local environment and run "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    print("Setup check passed.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "setup-check":
        setup_check()
        return

    if args.command == "train":
        train_model(
            project_root=PROJECT_ROOT,
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=args.name,
        )
        return

    if args.command == "eval-video":
        evaluate_video(
            weights=Path(args.weights),
            video_path=Path(args.video),
            conf=args.conf,
            show=not args.no_window,
            save=args.save,
            output_path=Path(args.output),
        )
        return

    if args.command == "all":
        weights = train_model(
            project_root=PROJECT_ROOT,
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=str(PROJECT_ROOT / "runs" / "detect"),
            name="object_train",
        )
        evaluate_video(
            weights=weights,
            video_path=PROJECT_ROOT / "colortest.mp4",
            conf=args.conf,
            show=not args.no_window,
            save=args.save,
            output_path=PROJECT_ROOT / "runs" / "video_eval" / "colortest_annotated.mp4",
        )
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
