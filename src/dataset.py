from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class DatasetReport:
    image_count: int
    label_count: int
    object_count: int


def validate_dataset(project_root: Path) -> DatasetReport:
    images_dir = project_root / "images"
    labels_dir = project_root / "labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images folder: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing labels folder: {labels_dir}")

    images = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    labels = sorted(p for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt")

    if not images:
        raise ValueError(f"No supported images found in {images_dir}")
    if not labels:
        raise ValueError(f"No .txt labels found in {labels_dir}")

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}
    missing_labels = sorted(image_stems - label_stems)
    extra_labels = sorted(label_stems - image_stems)

    if missing_labels:
        raise ValueError(f"Missing label files for images: {', '.join(missing_labels)}")
    if extra_labels:
        raise ValueError(f"Labels without matching images: {', '.join(extra_labels)}")

    object_count = 0
    for label_path in labels:
        for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"{label_path}:{line_number} must have 5 values: class x_center y_center width height"
                )

            try:
                class_id = int(parts[0])
                values = [float(value) for value in parts[1:]]
            except ValueError as exc:
                raise ValueError(f"{label_path}:{line_number} contains a non-numeric YOLO value") from exc

            if class_id != 0:
                raise ValueError(f"{label_path}:{line_number} has unsupported class id {class_id}; expected 0")
            if any(value < 0.0 or value > 1.0 for value in values):
                raise ValueError(f"{label_path}:{line_number} has coordinates outside the normalized 0..1 range")
            if values[2] <= 0.0 or values[3] <= 0.0:
                raise ValueError(f"{label_path}:{line_number} has non-positive width or height")

            object_count += 1

    if object_count == 0:
        raise ValueError("No labeled objects found.")

    return DatasetReport(
        image_count=len(images),
        label_count=len(labels),
        object_count=object_count,
    )


def write_dataset_yaml(project_root: Path) -> Path:
    config_dir = project_root / "configs"
    config_dir.mkdir(exist_ok=True)
    dataset_yaml = config_dir / "dataset.yaml"

    # Use an absolute path because Ultralytics resolves relative dataset paths
    # from the current process directory on some Windows setups.
    yaml_text = "\n".join(
        [
            f"path: {project_root.as_posix()}",
            "train: images",
            "val: images",
            "names:",
            "  0: object",
            "",
        ]
    )
    dataset_yaml.write_text(yaml_text, encoding="utf-8")
    return dataset_yaml
