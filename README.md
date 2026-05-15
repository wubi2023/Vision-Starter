# YOLO Object Dot Evaluation

This project trains a small YOLO object detector from the existing `images` and `labels` folders, then evaluates `colortest.mp4` by drawing a dot at the detected object center.

The original `images`, `labels`, and `colortest.mp4` files are not moved or modified.

## Setup

This project keeps Python packages inside a local virtual environment. On this machine, the working environment is `.venv312`, created from a project-local Python 3.12 runtime in `.python`.

```powershell
.\setup_env.ps1
.\.venv312\Scripts\Activate.ps1
```

If PowerShell blocks local scripts on your machine, run:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\setup_env.ps1
```

If you already created and activated a compatible environment manually:

```powershell
python -m pip install --no-cache-dir -r requirements.txt
```

## Commands

Validate the project:

```powershell
python main.py setup-check
```

Train with all labeled images:

```powershell
python main.py train
```

Short smoke-test training:

```powershell
python main.py train --epochs 1
```

Play the evaluation video with the dot overlay:

```powershell
python main.py eval-video
```

Play and save the annotated output video:

```powershell
python main.py eval-video --save
```

Train, then evaluate:

```powershell
python main.py all
```

## YOLO26 Commands

Validate the YOLO26 workflow:

```powershell
python main_yolo26.py setup-check
```

Train YOLO26 with all labeled images:

```powershell
python main_yolo26.py train
```

Short YOLO26 smoke-test training:

```powershell
python main_yolo26.py train --epochs 1
```

Train YOLO26 on a CUDA device:

```powershell
python main_yolo26.py train --device 0
```

Play the YOLO26 evaluation video with the dot overlay:

```powershell
python main_yolo26.py eval-video
```

Train YOLO26, then evaluate:

```powershell
python main_yolo26.py all
```

## Notes

- The detector uses class `0` as `object`.
- The current dataset is very small, so this is mainly a runnable framework and smoke test. Accuracy will improve with more labeled images.
- The video has no labels, so video evaluation is visual rather than mAP scoring.
