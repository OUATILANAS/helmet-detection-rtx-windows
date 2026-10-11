# Helmet detection — Windows RTX setup

Train on your **RTX 4060 Ti 8 GB / 16 GB RAM** PC, then bring candidate weights back to the Mac app. This repository contains the app source, grouped crop preparation, YOLO detector training, classifier training and strict full-frame evaluation. It does not promise 0.8 F1.

Start with **Windows 10/11 64-bit, Python 3.12 and native CUDA PyTorch**. WSL is optional, not required for this workflow. Run commands in **PowerShell**, from this repository's root. No paid cloud account is needed; local electricity and disk space are still used.

## 1. Install prerequisites

1. Install/update the [NVIDIA Windows driver](https://www.nvidia.com/Download/index.aspx), then restart if requested.
2. Install [Git for Windows](https://git-scm.com/downloads/win) and [Python 3.12, 64-bit](https://www.python.org/downloads/). With Windows Package Manager you can use:

```powershell
winget install --id Git.Git -e
winget install --id Python.Python.3.12 -e
```

Reopen PowerShell, then:

```powershell
git clone https://github.com/OUATILANAS/helmet-detection-rtx-windows.git
cd helmet-detection-rtx-windows
nvidia-smi
powershell -NoProfile -ExecutionPolicy Bypass -File windows/setup.ps1
```

The repository is public. You can clone it or download its source ZIP without signing in.

Setup installs CUDA PyTorch **2.9.1**, torchvision **0.24.1**, and Ultralytics **8.4.171** in `.venv`. PyTorch wheels include the CUDA runtime; a separate CUDA Toolkit installation is unnecessary for these commands. The setup checks CUDA availability and a real GPU backward pass. It stops if CUDA is unavailable. Version pins follow the [official Windows PyTorch installation table](https://pytorch.org/get-started/previous-versions/).

Use `.venv\Scripts\python.exe` directly; virtual-environment activation is unnecessary. Keep at least **15 GB free** for dependencies, data, checkpoints and outputs.

## 2. Get the current data and weights

Download **all three `helmet-windows-transfer.zip.part*` files** from this repository's **[Windows transfer release](https://github.com/OUATILANAS/helmet-detection-rtx-windows/releases/tag/windows-transfer-v1)**. The approximately 2.2 GB archive is split to fit GitHub's per-file limit. It contains:

- `dataset/helmet/`: materialized EdgeVision images and labels, with the existing train/val/test split and portable YAML. No Mac symlinks or absolute Mac paths. Mathematically zero-area source boxes are removed and recorded in `LABEL-CLEANING.json`; valid annotations and split assignments stay unchanged.
- `models/`: the eight checkpoints referenced by the Windows app profile, including the starting helmet localizer and classifier.
- `evaluation/`: the same frozen 16 fully reviewed validation frames used for previous comparisons, with rewritten relative paths and unchanged image bytes/labels.
- `TRANSFER-MANIFEST.json`: checksums and origin information.

Join the files, then extract **into the repository root**, preserving these folders. The join script verifies every part and the whole archive while reading in small chunks. The bundle is separate from Git history. No user-provided test screenshots are used for training. Test images stay in the test split; do not merge them into training.

```powershell
.venv\Scripts\python.exe training/join_transfer.py --parts "$env:USERPROFILE\Downloads"
Expand-Archive -Path helmet-windows-transfer.zip -DestinationPath . -Force
.venv\Scripts\python.exe training/verify_transfer.py
```

Dataset and model provenance is documented in [data attribution](docs/DATA-ATTRIBUTION.md) and [model sources](models/README.md). Third-party terms apply to their data and checkpoints.

## 3. Precheck before using hours of training time

```powershell
.venv\Scripts\python.exe training/check_cuda.py
.venv\Scripts\python.exe training/rtx_train.py preflight --imgsz 768 --batch 2
```

Preflight audits train/validation labels and exact duplicate separation, runs one short epoch on up to four real images per split, checks finite losses/weights, saves and reloads a checkpoint, and runs inference. It records GPU memory and duration in `data/outputs/rtx-preflight.json`. **Its temporary trained model is discarded; it is not the candidate.** Test images are not opened.

On 8 GB VRAM start at **768 pixels, batch 2, mixed precision**, `workers=0`, disk loading and no RAM cache. If you get CUDA out-of-memory, close GPU-heavy apps and retry with `--batch 1`; if necessary use `--imgsz 640 --batch 1`. Run preflight again after changing batch, resolution, weights or training data. Fit depends on the actual model; these are starting settings, not a memory guarantee.

## 4. Train a separate detector candidate

Only after a passing CUDA preflight:

```powershell
.venv\Scripts\python.exe training/rtx_train.py train --imgsz 768 --batch 2 --epochs 30 --name helmet-rtx-768
```

This fine-tunes `models/helmet_localizer_video_20261010.pt` on protective-helmet/no-helmet boxes. Backbone weights are trainable. It uses AMP, AdamW, modest learning rate, validation early stopping and checkpoint saving every epoch. It never overwrites app models. Keep the PC awake and plugged in.

Outputs:

- `runs/helmet-rtx-768/weights/best.pt`: best validation checkpoint.
- `runs/helmet-rtx-768/weights/last.pt`: interruption/resume checkpoint.
- `runs/helmet-rtx-768/results.csv`: losses and detector metrics by epoch.

If interrupted before the configured training completes:

```powershell
.venv\Scripts\python.exe training/rtx_train.py resume --checkpoint runs/helmet-rtx-768/weights/last.pt
```

Resume retains optimizer state and the original run's settings. Do not move the repository/data during a run. To start a further fine-tune after a completed run, use `--weights` pointing to its checkpoint, a new `--name`, and run a new preflight. This is a new run, not an optimizer-state resume. See [Ultralytics checkpoint documentation](https://docs.ultralytics.com/modes/train/).

Do not launch three models at once on this PC. Run one experiment, compare it, then choose the next. Measure the first full epoch before estimating the total time.

## 5. Compare the result honestly

First compare starting and candidate detectors using exactly the same frozen annotations and settings:

```powershell
.venv\Scripts\python.exe training/rtx_evaluate.py --candidate runs/helmet-rtx-768/weights/best.pt --imgsz 768 --confidence 0.25 --output data/outputs/detector-comparison
```

Then compare them inside the complete app, replacing only the candidate's localized head model:

```powershell
.venv\Scripts\python.exe training/rtx_evaluate.py --pipeline-config configs/windows-rtx.yaml --candidate runs/helmet-rtx-768/weights/best.pt --output data/outputs/pipeline-comparison
```

The evaluator reports helmet/no-helmet macro F1 for all frames, video instants and still images. Missing heads, abstentions and false positives count. These fixed 16 frames are **validation**, not new independent test evidence. GPU results may differ slightly from prior CPU results; re-run both baseline and candidate on the same PC. Detector-only scores and complete-app scores have different meanings.

After choosing a candidate and locking thresholds, evaluate the detector's held-out image test split once:

```powershell
.venv\Scripts\python.exe training/rtx_train.py test --checkpoint runs/helmet-rtx-768/weights/best.pt --imgsz 768 --batch 2
```

That command reports YOLO mAP; **mAP is not the strict helmet F1**. Broad unseen-video F1 still needs a fully reviewed independent manifest. The current 56-frame broad queue is only 29/56 reviewed and is deliberately excluded from this transfer. Do not select checkpoints on test data. See [current benchmark findings](docs/LATEST-BENCHMARK.md).

## 6. Optional classifier fine-tuning

Keep detector and classifier changes separate so their effects can be measured:

```powershell
.venv\Scripts\python.exe training/finetune_head_classifier.py --dataset dataset/helmet --initial models/helmet_head_edgevision.pt --output data/outputs/classifier-rtx --device cuda --epochs 12 --batch-size 16 --learning-rate 0.00002 --skip-test
```

This classifier measures head-crop quality, not full-frame detection. It uses small head crops and should be tested separately before app integration. It does not offer optimizer-state resume. The default detector path above is the first experiment; changing both at once makes comparisons harder.

## 7. Run the app and bring weights back to the Mac

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File windows/run-app.ps1
```

Open **http://127.0.0.1:8013/**. The Windows profile uses the GPU for detection and CPU for crop classification. Initial models and images process locally. The app also provides manual annotation workflows; inspect them before exporting new training data.

Copy the candidate `best.pt`, comparison `report.json`, `results.csv`, and preflight report back to the Mac. Keep the original model as rollback. Candidate production promotion requires improvement across still/video metrics, false-positive checks, non-rider checks and motorcycle-absent frames; reaching 0.8 on cropped heads alone does not qualify.

## Troubleshooting

- **CUDA false / CPU wheel:** update the NVIDIA driver, rerun setup, and inspect `nvidia-smi` plus `training/check_cuda.py`.
- **Out of memory:** lower batch first, then resolution; rerun preflight with the exact new settings. Lower resolution may reduce small-head recall.
- **RAM pressure:** keep `cache=False`, `workers=0`, and close other applications. The dataset loads from disk.
- **Missing model/data:** check that the release ZIP was extracted into the repo root and run `training/verify_transfer.py`.
- **Port 8013 busy:** stop the existing app, or run uvicorn with another `--port`.
- **Windows multiprocessing errors:** the provided entrypoints use a main guard and zero loader workers.
- **RF-DETR/RT-DETR experiments:** do not copy the old Mac-specific pilot scripts blindly. Their native-scale loaders, checkpoint formats and backend checks need a separate CUDA port. This repository's first supported training path is YOLO; it does not claim those experiments are already GPU-validated.

## Verification

`python -m pytest` runs the model-free app tests and portable-workflow tests. Repository checks were run on the Mac; Windows PowerShell and CUDA execution must be verified on your PC using the provided setup/preflight. No Windows GPU access is assumed.
