# Helmet Rider Detection

A local, two-stage computer-vision project for checking helmet use by people
who are geometrically associated with a motorcycle or bicycle. Pedestrians are
not classified. Image and video pixels are processed locally; no identities,
face embeddings, or cloud inference are used.

## State of this first implementation

The project code, CLI, API, training workflow, and automated logic tests are
included. Checkpoint files stay local and are not committed. The desktop profile
uses a traffic-tuned motorbike detector and a separate helmet/no-helmet detector.
Helmet status is attached to a rider only when a helmet box overlaps that
associated rider's head region; riders without clear evidence remain UNKNOWN.

## Architecture

    Camera / image / video
             |
             v
    YOLO26n person + bicycle  |  TrafficAI motorbike detector
              \___________ helmet / no-helmet detector __________/
             |
             v
    Rider/vehicle geometric association
       |                             |
       | insufficient evidence       +--> vehicle and rider IDs
       +--> ignore pedestrian                    |
                                                  v
                                      head / upper-head ROI
                                                  |
                                                  v
                                      head-region helmet match
                                                  |
                                                  v
                                   helmet / no-helmet status
                                                  |
                                                  v
                                        image / UI / JSON

The associator uses lower-body overlap with the vehicle, person overlap,
horizontal alignment, and bottom-center proximity. Expanded vehicle geometry
adds context but cannot establish an association on its own. Each person can
map to one vehicle; one vehicle can map to multiple riders. Helmet model boxes
are assigned only to riders with a helmet box overlapping the estimated head
region. Riders with no matched evidence remain UNKNOWN.

The detector and classifier are interfaces behind separate implementations.
The desktop profile uses YOLO26n for people and bicycles, with a second
motorbike-specific YOLOv8 model from the [TrafficAI project](https://github.com/VinhTh-x6/TrafficAI)
and a helmet/no-helmet YOLOv8 detector from [sharathhhhh on Hugging Face](https://huggingface.co/sharathhhhh/safetyHelmet-detection-yolov8).
The helmet detector uses a 2×2 overlapping tile pass to retain small, distant
heads, alongside a full-frame pass. Tiled boxes are mapped back to the original
image and deduplicated before they are matched to rider head regions. The tile
confidence cutoff is 0.005; detections are counted only when they overlap an
associated rider's head region. When both the person box and helmet box have low
confidence, the helmet evidence is ignored and the rider remains UNKNOWN. This
helps avoid vehicle details such as mirrors being labeled as helmets while
preserving stronger helmet detections on weak person boxes. Ambiguous matches
use both head-region overlap and helmet confidence, so a weak nearby box does not
displace a clearer helmet detection.
That model was fine-tuned for dense Vietnamese traffic and detected 132
motorbikes in the supplied crowded-road image during local evaluation. Its
checkpoint was trained at 960 pixels; this app uses a 1280-pixel input, a 0.04
motorbike confidence cutoff, 0.45 NMS IoU, and test-time augmentation. The
YOLO26n people and bicycle detections keep a separate 0.08 confidence cutoff.
This improves recall in crowded scenes at the cost of inference time and can
still produce false positives. The Pi profile keeps the smaller single-model
ONNX Runtime path for faster CPU inference; it does not load the desktop helmet
checkpoint.
NCNN remains available through the Ultralytics wrapper and may be faster on a
particular Pi. YOLO11n is another fallback if a target runtime behaves better.
Actual latency and accuracy depend on camera resolution, image size, scene,
runtime build, and thermal state. Run the benchmark on the target device
instead of extrapolating desktop results.

## Project layout

    configs/                 desktop and Raspberry Pi profiles
    models/                  ignored local model checkpoints
    src/helmet_rider/         typed detector, association, ROI, tracking, pipeline
    app/                      FastAPI API and vanilla HTML interface
    scripts/                  image, camera, video, benchmark, model export
    training/                 dataset prep, classifier training and evaluation
    dataset/                  local sources, prepared splits, and evaluation data
      collected-training-data/ index of the datasets and review clips used
    tests/                    model-free association, ROI, pipeline, tracker tests
    deploy/                   optional systemd service template

## Install

Python 3.10 or newer is required; Python 3.11 is a practical choice on current
Raspberry Pi OS. From this project directory:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -e ".[dev,ultralytics]"

The Ultralytics development dependency includes PyTorch. It is not required
for direct ONNX Runtime inference. The classifier has no torchvision
requirement and is deliberately small. No NVIDIA or CUDA packages are needed.

For optional ONNX export and direct ONNX Runtime inference on a development
machine, install the separate extra with pip install -e ".[onnx]". The Pi
requirements install only the ONNX Runtime extra. See the [ONNX Runtime Python
installation guide](https://onnxruntime.ai/docs/get-started/with-python.html)
for supported CPU packages, including Linux ARM64.

## Install detector weights and export

The base detector uses COCO class IDs for person, bicycle, and motorcycle. To export
ONNX on the development machine, install the onnx extra. Explicitly
download the official nano checkpoint and export an ONNX model:

    python scripts/export_models.py --kind detector --model yolo26n.pt \
      --format onnx --imgsz 320 --device cpu

The exporter places files next to the source model. Copy yolo26n.onnx to
models/yolo26n.onnx for the Pi profile, or keep the .pt file at
models/yolo26n.pt for the desktop profile. Run model acquisition/export while
online; normal inference only reads local files. ONNX export enables embedded
NMS for the direct ONNX Runtime parser. Export support also includes NCNN and
LiteRT (the current Ultralytics name for the TFLite family). For NCNN, install
the Ultralytics extra and set detector.backend to ncnn. Use --verify with an
image to compare exported detector box overlap with the source checkpoint;
classifier ONNX export performs a numerical logits check when ONNX Runtime is
installed. Numerical parity of every detector/runtime combination has not
been measured in this repository.

Ultralytics documents YOLO26 support and Raspberry Pi 5 setup at
[its Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi),
and NCNN export at
[its NCNN integration guide](https://docs.ultralytics.com/integrations/ncnn).
Review model/library licensing for the intended distribution and use.

The desktop profile sets `detector.device: auto`: it selects CUDA when
available, Apple Metal (MPS) on supported Macs, and CPU otherwise. The
Raspberry Pi profile remains explicitly CPU-based. MPS uses PyTorch's CPU
fallback for operations such as NMS that are not implemented on the Metal
backend.

The default desktop profile also needs `models/trafficai_motorbike.pt`. Download
the `best.pt` checkpoint from [VinhTh-x6/TrafficAI](https://github.com/VinhTh-x6/TrafficAI/tree/main/models)
and save it under that name. The project lists an MIT license. This checkpoint
was trained on Vietnamese vehicle images, so verify it on the camera angles and
traffic conditions used in your deployment.

It also needs `models/safety_helmet_yolov8.pt`, the `best.pt` checkpoint from
[sharathhhhh/safetyHelmet-detection-yolov8](https://huggingface.co/sharathhhhh/safetyHelmet-detection-yolov8).
The model card lists Apache-2.0 and `with_helmet` / `without_helmet` classes.
Both local model sources and their licensing notes are recorded in
[models/README.md](models/README.md).

## Image inference

After model files have been installed:

    python scripts/run_image.py --source path/to/traffic.jpg --save --json
    python scripts/run_image.py --source path/to/traffic.jpg --model models/yolo26n.onnx --backend onnxruntime \
      --imgsz 320 --confidence 0.28 --show --debug

The default command writes data/outputs/annotated.png and data/outputs/result.json.
Use --save path.png or --json path.json to choose output locations. Saving is
off unless requested. The JSON result lists vehicles, associated riders,
confidence, head ROI boxes, timings, and summary counts.

## Live camera and video

The web app includes a **Video file** tab for recorded CCTV clips. Choose an
MP4, M4V, MOV, or WebM file and start video analysis. The clip plays in your
browser; the app sends sampled still frames to its local detector as playback
continues. Analysis runs at the speed of the selected model and hardware.

For a live IP camera, choose **Live camera → Network CCTV · RTSP** and enter the
camera's RTSP or RTSPS stream URL, for example
`rtsp://192.168.1.40:554/Streaming/Channels/101`. The app host must be able to
reach the camera on the local network. A username and password can be included
in the URL if the camera requires them. The stream itself is processed on the
app host; use a secured network and avoid sharing the URL.

OpenCV webcam (optional local preview; q quits):

    python scripts/run_camera.py --camera 0 --show

Raspberry Pi Camera:

    python scripts/run_camera.py --source picamera --config configs/raspberry_pi.yaml

Save a camera recording only when requested:

    python scripts/run_camera.py --camera 0 --save data/outputs/camera.mp4

Video input and optional annotated output:

    python scripts/run_video.py --source path/to/traffic.mp4 --show --save

Detector frame interval, classifier reclassification interval, camera size,
frame skipping, model input size, tracking age, and smoothing window are
configurable in YAML. The IoU tracker associates adjacent detections without
appearance features or identity persistence.

## Lightweight web interface

    uvicorn app.api:app --host 0.0.0.0 --port 8000

The selected desktop head refinement model is documented in
[the evaluation report](data/outputs/head-detector-20261010/REPORT.md).
The existing 78-image head benchmark improved from 0.547 to 0.827 macro F1;
an untouched 40-frame video holdout improved from 0.223 to 0.445 and remains
below the image target. A larger frozen comparison on 306 other images improved
0.689 to 0.761; 393 frames from ten new clips improved 0.342 to 0.524.

<!-- video-gap-experiment-20261010 -->
A newer experimental YOLOv8-P2 head pipeline was selected on 140 validation
frames and tested after freezing its model/config hashes. On the exact original
40-frame test it improved 0.445 to 0.615; eight reserved clips (240 frames,
676 heads) improved 0.613 to 0.808. A deterministic 40-image comparison improved
0.646 to 0.655. These publisher-label metrics omit unmatched background false
positives: adding 13 visually confirmed backgrounds on an eight-frame audit
reduces that subset's F1 from 0.870 to 0.733. The crowded-photo regression also
fails on an additional helmeted rider. The experiment is **not promoted**;
consistent 0.8+ and motorcycle/no-motorcycle accuracy remain unproven. See the
[complete experiment report](data/outputs/video-gap-20261010/REPORT.md).
<!-- /video-gap-experiment-20261010 -->

The broader 0.8–0.9 target remains unmet. See
[the expanded evaluation](data/outputs/head-detector-20261010/expanded-evaluation/REPORT.md)
for source-specific results, overlap checks and failure examples.
These metrics count misses and abstentions, but publisher
annotations do not exhaustively score background false positives. The desktop
defaults load the additional fine-tuned head localizer and crop classifier.
On this Mac, use the verified native runtime:

    PYTHONPATH=src .venv-training/bin/python -m uvicorn app.api:app --host 127.0.0.1 --port 8013

Open http://localhost:8000 on the host, or use the host's LAN address from a
device on the same network. The page supports image uploads, the browser camera
on the computer viewing the page, and a USB webcam or Pi Camera attached to the
app host. Connecting the browser camera starts continuous live analysis by
sending sequential JPEG snapshots to the local API; inference never queues
frames faster than it can process them. Pause analysis to keep the preview live,
or stop the camera to release it. Browser camera access works on localhost or
over HTTPS and requires the user to grant permission. App-host camera streams
also analyze frames continuously; without detector weights they show a preview
only. The API exposes:

    POST /api/detect
    POST /api/camera/stop
    GET  /api/camera/stream?source=webcam
    GET  /api/camera/stream?source=picamera
    GET  /api/health

For Pi web use, select the Raspberry Pi profile when starting the service:

    HELMET_RIDER_CONFIG=configs/raspberry_pi.yaml \
      uvicorn app.api:app --host 0.0.0.0 --port 8000

The web page is intentionally plain HTML and JavaScript. It does not load a
remote font, framework, or analytics script.

## Training and dataset preparation

See [training/README.md](training/README.md) for sequence-aware splitting and
the complete workflow. Provide head crops with labels in helmet and no_helmet
classes. Use a CSV path,label,sequence manifest for video-derived samples so
frames from a source video remain in one split. Do not automatically download
datasets.

Training is for a development machine, not the Pi:

    python training/prepare_dataset.py --source /path/to/crops \
      --annotations /path/to/annotations.csv --output dataset --clean
    python training/train_classifier.py --dataset dataset --output models/training \
      --epochs 30 --batch-size 32 --size 128 --seed 17
    python training/evaluate_classifier.py --dataset dataset \
      --checkpoint models/training/best.pt --split test
    cp models/training/best.pt models/helmet_classifier.pt

The dataset tool checks corrupt files, groups sequence IDs, creates train,
validation, and test folders, and emits class counts. Training uses weighted
cross-entropy and mild image-quality augmentations. Evaluation reports accuracy,
per-class precision, recall, F1, support, macro metrics, and confusion matrix.
Record any chosen confidence thresholds against a validation set.

### Review motorcycle detections in the web app

After analyzing an image or a captured PC camera frame, use **Correct motorcycle
boxes** to select and remove false detections or draw boxes around missed
motorcycles. Save each checked frame; empty labels are valid when a reviewed
frame contains no motorcycles. Saved originals, YOLO labels, and metadata stay
locally under `dataset/motorcycle_review/` (this directory is git-ignored).

Use the same capture group name for frames from one video or scene, and a new
name for an independent scene. Once at least two groups are saved, choose
**Download YOLO dataset**. The ZIP contains a YOLO dataset configuration and a
group-disjoint train/validation/test split when there are at least three groups
(two groups produce train and validation). Fine-tune from the local
traffic-specific motorbike checkpoint, then compare on the held-out test split:

    yolo detect train model=models/trafficai_motorbike.pt \
      data=dataset.yaml imgsz=1280

The labels are motorcycle-only; this workflow improves motorcycle localization
and does not retrain the helmet detector or rider association logic.

The compact classifier checkpoint can currently be exported to ONNX with a
numerical PyTorch/ONNX Runtime check:

    python scripts/export_models.py --kind classifier --model models/helmet_classifier.pt \
      --format onnx --classifier-size 96 --output models/helmet_classifier.onnx

The custom classifier's NCNN and LiteRT paths still require runtime-specific
conversion and parity benchmarking. The project does not install these extra
runtimes by default.

## Configuration and debug mode

Edit configs/default.yaml for development or configs/raspberry_pi.yaml for the
Pi starting profile. Important controls include:

- detector confidence, input size, and every-N-frame inference
- association score and lower-body contact threshold
- head ROI height, width expansion, and minimum size
- helmet confidence thresholds and reclassification interval
- IoU track age and temporal smoothing history
- camera width, height, FPS, and frame skipping
- debug geometry and association lines

Pass --debug in a CLI or set visualization.debug: true in YAML to show raw
person boxes, vehicle association regions, bottom-center points, candidate
scores, and pipeline time. It never changes which people can be classified.

## Raspberry Pi 5 deployment

Use a 64-bit Raspberry Pi OS installation. Install system camera and compiler
packages, then create a virtual environment that can see Picamera2's OS Python
bindings:

    sudo apt update
    sudo apt install -y python3-venv python3-picamera2 libopenblas0
    cd ~/helmet-rider-detection
    python3 -m venv --system-site-packages .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements-pi.txt

Picamera2 is provided by Raspberry Pi OS and should not be pip-installed into
this environment. The --system-site-packages option makes the apt-installed
picamera2 module visible inside the venv. The Pi requirements install ONNX
Runtime without Ultralytics or PyTorch. OpenCV is installed from the project
requirements; verify its import and camera capture after setup.
Do not add CUDA or NVIDIA packages.

Copy the exported detector to models/yolo26n.onnx and the classifier to
models/helmet_head_edgevision.onnx. Benchmark
detector sizes 320, 416, and 512 on the actual Pi and choose the best acceptable
accuracy/latency tradeoff:

    python scripts/benchmark.py --source picamera \
      --config configs/raspberry_pi.yaml --iterations 100 --warmup 10 --report
    python scripts/run_camera.py --source picamera --config configs/raspberry_pi.yaml
    HELMET_RIDER_CONFIG=configs/raspberry_pi.yaml \
      uvicorn app.api:app --host 0.0.0.0 --port 8000

Use a static image containing bikes/riders for repeatable benchmarking; a
pedestrian-only image will not exercise the helmet stage. A source such as
camera:0 or picamera includes capture timing. Benchmark repeated representative
scenes and report measured hardware, OS, model, size, FPS, latency, and memory.
The benchmark script writes Markdown only when --report is supplied. The
repository contains no Pi measurement.

## Optional systemd service

Edit deploy/helmet-rider-detection.service to match your user and checkout,
then install it:

    sudo cp deploy/helmet-rider-detection.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable --now helmet-rider-detection
    systemctl status helmet-rider-detection

The service starts the local API at boot and is optional.

## Tests and troubleshooting

Run the model-free logic suite:

    pytest

The tests cover rider association, parked-bike bystanders, one-to-many riders,
separate vehicles, bicycle association, edge crops, and the guarantee that
pedestrians do not invoke the helmet classifier. No model is required for
these tests.

Common issues:

- Detector model not found: install a local .pt checkpoint or exported directory
  at the path configured in YAML.
- All riders say UNKNOWN: train and install the classifier checkpoint. No
  default classifier is bundled.
- No rider associated: enable debug mode and inspect the geometric candidate
  score and lower-body overlap before adjusting the association threshold.
- Camera permission/open failure: check operating-system access and camera
  index; close other processes holding the webcam.
- Picamera2 import failure in the venv: recreate it with
  --system-site-packages and install python3-picamera2 through apt.
- Slow Pi inference: use NCNN, lower model input size, lower camera resolution,
  and increase detector/classifier frame intervals. Measure each change.
- Unexpected classification: inspect head crops and sequence split quality,
  collect representative examples, evaluate per-class metrics, and use UNKNOWN
  for uncertain results.

## Limitations

The generic COCO detector may miss small or occluded bikes, may not separate
closely packed people, and does not know whether a detected bike is parked.
Association is a geometric heuristic; without pose or temporal motion features
it cannot reliably resolve every person standing directly over a bicycle or
motorcycle. The initial head ROI is a configured top slice of the person box.
The helmet classifier has no pretrained weights and depends on the quality,
coverage, and labeling of the training data. Verify each target camera and
deployment profile with representative annotated examples before use.

## Accuracy improvements and review workflow (October 2026)

The desktop profile enables `classifier.verify_small_heads`: an unknown head can
become no-helmet only when a localized bare-head detection and two crop views
agree, with supported motorcycle geometry. Streams require two fresh classifier
passes. The 2026-10-10 check recovered four additional published no-helmet heads
on 78 traffic images; the sampled video outputs were unchanged. This is a limited
dataset regression result, not an estimate for a new camera. Set the flag to
`false` to retain the previous behavior. Details are in
`data/outputs/crop-policy-20261010/REPORT.md`.

Stream decisions now require two fresh classifier confirmations for NO HELMET.
CLI camera/video and the web app share this rule. Browser playback timestamps
and live capture times control motion prediction and evidence expiry. Seeking,
changing frame dimensions, or a long capture gap resets temporal state. Results
include evidence ages, detector scores, classifier probabilities, crop size,
and confirmation counts. Scores from separate models are not interchangeable
calibrated probabilities.

Video/camera detail defaults to original source resolution with JPEG quality
0.97. Choose 1280 or 640 px for smaller transfers. The desktop host-camera
profile requests 1920 × 1080; the actual resolution depends on the camera.
Model input resolution remains a separate setting.

The existing footage has 13 original-resolution frames in the **Local footage
review queue**. Open a frame, correct motorcycle/head boxes, remove non-riders,
add missed heads, and explicitly label Helmet / No helmet / Unsure. Mark a frame
fully reviewed only after including all visible rider heads. Optional rider
references identify the same rider across frames for temporal evaluation;
model-generated track IDs are not human identity labels. Select a reviewed
motorcycle link to measure association correctness. Keep all frames from a clip
in the capture group supplied by the queue.

Audit and score the complete rider pipeline:

    python scripts/evaluate_riders.py --audit-only
    python scripts/evaluate_riders.py --output data/outputs/rider-evaluation
    python scripts/evaluate_riders.py --temporal --tracker iou
    python scripts/evaluate_riders.py --temporal --tracker bytetrack

ByteTrack is an experimental alternative (`tracking.backend: bytetrack`),
installed with `pip install -e ".[tracking]"`. The default remains the tracker
that uses elapsed time and IoU. Event scores require human rider references
and source timestamps; absent labels are reported as unavailable. UNKNOWN and
missed riders reduce end-to-end recall. Unmatched predictions count as false
positives only on explicitly fully reviewed frames. Duplicate images with
conflicting labels or groups are excluded from evaluation and block export.

Prepare and fine-tune the active EfficientNet classifier once enough independent
human labels exist:

    python training/prepare_runtime_reviews.py --output dataset/runtime-head-reviews-v2
    python training/finetune_head_classifier.py --dataset dataset/runtime-head-reviews-v2 \
      --initial models/helmet_head_edgevision.pt --output data/outputs/runtime-candidate

Both classes are required in train, validation, and test. The runtime manifest
preserves the original model crop alongside the corrected human head box.
Crops that miss most of the reviewed head are excluded from binary training.
Training adds crop offsets, blur, low-resolution appearance, and JPEG artifacts.
Checkpoint selection uses validation; test runs after selection. Unknowns and
non-head objects are not NO HELMET training labels. The current reviews do not
meet the training requirements.

Fit exploratory temperature and thresholds on validation crops only:

    python training/calibrate_classifier.py --dataset dataset/edgevision/yolo \
      --checkpoint models/helmet_head_edgevision.pt --output data/outputs/calibration

Apply candidate settings through `classifier.temperature` and class thresholds
in a separate YAML configuration, then evaluate final rider decisions. This
crop-level calibration does not validate camera-level accuracy. Matched head
crops are also an isolated option (`head_roi.strategy: matched`). The default
`verified` strategy keeps the person crop primary and only confirms uncertain
helmet cases when a strong detector and sufficiently detailed localized head
classification agree. See `data/outputs/localization-20261010/REPORT.md` for
the limited photo validation and annotation-box matching caveats. Set the
strategy to `person` to disable this additional positive verification.

Measure one-factor performance candidates:

    python scripts/ablate_pipeline.py --iterations 3 --warmup 2

Each source/variant is warmed before measurement. Output includes stage p50/p95,
reviewed status metrics, crowded-scene motorcycle matches, and unlabeled outputs.
Fewer UNKNOWN labels on an unlabeled frame do not establish better accuracy.
`classifier.device` supports CPU, auto, MPS, and CUDA. The desktop profile now uses auto after CPU/MPS crop and pipeline parity checks;
it falls back to CPU when no accelerator is available.

The Pi configuration now uses the exported EfficientNet checkpoint with its
correct 224 px padding/normalization. Export it with:

    python scripts/export_models.py --kind classifier --classifier-backend efficientnet_b0 \
      --classifier-size 224 --model models/helmet_head_edgevision.pt \
      --format onnx --output models/helmet_head_edgevision.onnx

Desktop export/runtime parity is checked separately from Pi speed or accuracy.
The Pi still uses a small generic detector and needs its own device benchmark.
