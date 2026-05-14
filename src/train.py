from __future__ import annotations

import os
from pathlib import Path

from src.dataset import validate_dataset, write_dataset_yaml


def train_model(
    project_root: Path,
    model_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    project: str,
    name: str,
) -> Path:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(project_root / ".ultralytics"))

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Activate the local environment and run "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    report = validate_dataset(project_root)
    data_yaml = write_dataset_yaml(project_root)

    print(f"Training on {report.image_count} images and {report.object_count} labeled objects.")
    model = YOLO(model_name)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        exist_ok=True,
        seed=42,
        device=None,
    )

    save_dir = Path(getattr(results, "save_dir", Path(project) / name))
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Training completed, but best weights were not found at {best_weights}")

    print(f"Best weights: {best_weights}")
    return best_weights
