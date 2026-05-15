from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.dataset import validate_dataset, write_dataset_yaml
from src.train import train_model
from src.video_eval import evaluate_video


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "yolo26n.pt"
DEFAULT_RUN_NAME = "object_train_yolo26"
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))


def is_cuda_like_device(device: str | None) -> bool:
    if device is None:
        return False

    value = device.strip().lower()
    if value in {"cuda", "gpu"} or value.startswith("cuda:"):
        return True

    first_value = value.split(",", 1)[0]
    return first_value.isdecimal()


def first_cuda_index(device: str) -> int:
    value = device.strip().lower()
    if value in {"cuda", "gpu"}:
        return 0
    if value.startswith("cuda:"):
        value = value.split(":", 1)[1]
    value = value.split(",", 1)[0]
    return int(value)


def validate_cuda_device(device: str | None) -> None:
    if not is_cuda_like_device(device):
        return

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is not installed. Activate the local environment and install GPU-enabled torch."
        ) from exc

    if not torch.cuda.is_available():
        if torch.version.cuda is None:
            detail = "Installed PyTorch is CPU-only. Install a CUDA-enabled PyTorch build."
        else:
            detail = (
                "Installed PyTorch includes CUDA, but the NVIDIA driver/device is not available "
                "to this process. Check `nvidia-smi` and `/dev/nvidia*` before training."
            )
        raise SystemExit(
            "CUDA GPU was requested but is not available to PyTorch.\n"
            f"Requested device: {device}\n"
            f"Installed torch: {torch.__version__}\n"
            f"torch.version.cuda: {torch.version.cuda}\n"
            f"{detail}"
        )

    device_index = first_cuda_index(device or "0")
    device_count = torch.cuda.device_count()
    if device_index < 0 or device_index >= device_count:
        raise SystemExit(f"CUDA device {device_index} is not available. Found {device_count} CUDA device(s).")

    print(f"CUDA device {device_index} is available: {torch.cuda.get_device_name(device_index)}")
    print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the YOLO26 detector with a center-dot video overlay."
    )
    device_parser = argparse.ArgumentParser(add_help=False)
    device_parser.add_argument(
        "--device",
        default=None,
        help="Optional device for training/inference, for example cpu, 0, cuda:0, or 0,1.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "setup-check",
        parents=[device_parser],
        help="Validate Python packages, dataset files, project config, and optional CUDA device.",
    )

    train_parser = subparsers.add_parser(
        "train",
        parents=[device_parser],
        help="Train a YOLO26 detector using all images and labels.",
    )
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--imgsz", type=int, default=640)
    train_parser.add_argument("--batch", type=int, default=4)
    train_parser.add_argument("--model", default=DEFAULT_MODEL)
    train_parser.add_argument("--project", default=str(PROJECT_ROOT / "runs" / "detect"))
    train_parser.add_argument("--name", default=DEFAULT_RUN_NAME)

    eval_parser = subparsers.add_parser(
        "eval-video",
        parents=[device_parser],
        help="Evaluate colortest.mp4 with YOLO26 and a center-dot overlay.",
    )
    eval_parser.add_argument(
        "--weights",
        default=str(PROJECT_ROOT / "runs" / "detect" / DEFAULT_RUN_NAME / "weights" / "best.pt"),
        help="Path to trained YOLO26 weights.",
    )
    eval_parser.add_argument("--video", default=str(PROJECT_ROOT / "colortest.mp4"))
    eval_parser.add_argument("--conf", type=float, default=0.5)
    eval_parser.add_argument("--save", action="store_true")
    eval_parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "runs" / "video_eval" / "colortest_annotated_yolo26.mp4"),
    )
    eval_parser.add_argument("--no-window", action="store_true", help="Do not open a playback window.")

    all_parser = subparsers.add_parser(
        "all",
        parents=[device_parser],
        help="Validate, train YOLO26, then evaluate the video.",
    )
    all_parser.add_argument("--epochs", type=int, default=100)
    all_parser.add_argument("--imgsz", type=int, default=640)
    all_parser.add_argument("--batch", type=int, default=4)
    all_parser.add_argument("--model", default=DEFAULT_MODEL)
    all_parser.add_argument("--conf", type=float, default=0.5)
    all_parser.add_argument("--save", action="store_true")
    all_parser.add_argument("--no-window", action="store_true")

    return parser


def setup_check(device: str | None) -> None:
    report = validate_dataset(PROJECT_ROOT)
    dataset_yaml = write_dataset_yaml(PROJECT_ROOT)
    validate_cuda_device(device)

    print(f"Images: {report.image_count}")
    print(f"Labels: {report.label_count}")
    print(f"Objects: {report.object_count}")
    print(f"Dataset config: {dataset_yaml}")

    try:
        import cv2  # noqa: F401
        import ultralytics  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing package: "
            f"{exc.name}. Activate the local environment and run "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    print("YOLO26 setup check passed.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "setup-check":
        setup_check(args.device)
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
            device=args.device,
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
            device=args.device,
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
            name=DEFAULT_RUN_NAME,
            device=args.device,
        )
        evaluate_video(
            weights=weights,
            video_path=PROJECT_ROOT / "colortest.mp4",
            conf=args.conf,
            show=not args.no_window,
            save=args.save,
            output_path=PROJECT_ROOT / "runs" / "video_eval" / "colortest_annotated_yolo26.mp4",
            device=args.device,
        )
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
