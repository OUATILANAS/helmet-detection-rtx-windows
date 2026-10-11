# Local detector weights

The selected desktop refinement checkpoints are
`helmet_localizer_video_20261010.pt` (YOLOv8 head detection) and
`helmet_head_video_20261010.pt` (EfficientNet-B0 crop classification).
They supplement the existing person, vehicle and head classifiers through
the localized refinement settings in `configs/default.yaml`.
Their JSON sidecars record hashes and training provenance. Actual pipeline
image head macro F1 improved 0.547 → 0.827; untouched video-frame head F1
improved 0.223 → 0.445, with partial publisher annotations. The photo checks
retain all five user-confirmed helmets and keep the red cap as no helmet.
See `data/outputs/head-detector-20261010/REPORT.md` for completed training,
evaluation limits, native runtime and rollback. Noncommercial source dataset
conditions apply; these are local research checkpoints.

`helmet_head_headwear_20261010.pt` remains rolled back as the primary classifier;
the selected pipeline uses it as a separate disagreement/consensus verifier.

`helmet_yolo26s_edgevision_20261010_experimental.pt` is the best-validation
checkpoint from a three-epoch YOLO26s training experiment on EdgeVision head
boxes. The run did not converge adequately and failed the crowded and occluded
photo checks. It is not active. See `data/outputs/model-benchmark-20261010/REPORT.md`
for validation metrics and reproduction details.

`traffic_helmet_2vhoc.pt` and `.onnx` are comparison candidates from
[2vhoc/helmet-detection-traffic](https://huggingface.co/2vhoc/helmet-detection-traffic),
revision `7e46b9388b05c497ed776026108b5ee7f08f0a41`, stage 2. The export identifies
YOLO26l trained on helmet/license-plate crops at 320 px, with classes helmet,
nohelmet and licenseplate. Both full-frame/tiled and person-crop tests failed
the user-confirmed red-cap regression; these checkpoints are not active.
The ONNX metadata specifies AGPL-3.0. Download provenance and benchmark results
are saved under `data/outputs/model-benchmark-20261010`.

The desktop profile reads `trafficai_motorbike.pt` for motorbike detection,
`yolo26n.pt` for people and bicycles, and `safety_helmet_yolov8.pt` for helmet
status. Checkpoint files are kept local and are excluded from source control.

`trafficai_motorbike.pt` is the `best.pt` checkpoint from
[VinhTh-x6/TrafficAI](https://github.com/VinhTh-x6/TrafficAI/tree/main/models).
That project lists an MIT license and describes its YOLOv8 model as trained on
Vietnamese traffic. The model was evaluated on the sample image supplied with
this project before it was selected. Review the upstream project and its data
sources before redistributing the checkpoint.

`visdrone-yolo11s.pt` is a YOLO11s checkpoint fine-tuned on VisDrone-DET from
[dronefreak/visdrone-yolo11s](https://huggingface.co/dronefreak/visdrone-yolo11s).
The app uses its `motor` class only as a recall fallback when the primary
detectors found fewer than 24 motorcycles. It runs on the full frame and adds
only confident, non-duplicate boxes above the configured minimum size. It
improved detection of a distant motorcycle group in the supplied street image;
it is not a replacement for the primary road-level detector and may still miss
small or occluded motorcycles. The model card specifies AGPL-3.0; review its
terms before distributing or exposing the app as a network service.

`safety_helmet_yolov8.pt` is `best.pt` from
[sharathhhhh/safetyHelmet-detection-yolov8](https://huggingface.co/sharathhhhh/safetyHelmet-detection-yolov8).
Its model card specifies `with_helmet` and `without_helmet` classes and an
Apache-2.0 license. Helmet boxes are associated to tracked riders only when
they overlap the rider's estimated head region.

`helmet_head_edgevision.pt` is an EfficientNet-B0 head-crop classifier initialized
from the Apache-2.0 [Helmet v5 CCTV project](https://huggingface.co/vivekvar/helmet-v5)
and fine-tuned on the training split of the EdgeVision dataset. Its class order
is helmet, no-helmet. This is now the retained starting/rollback checkpoint.
The classifier confirms helmet-detector evidence and classifies associated rider
heads; it does not replace the
vehicle, person, or helmet detector. On the held-out EdgeVision head-crop test
split it achieved 92.6% accuracy, with 95.5% helmet recall and 91.0% no-helmet
recall. Those crop metrics are domain-specific and are not a guarantee for
occluded or very small CCTV riders. The fine-tuning data attribution and license
are recorded in `dataset/edgevision/README.md`.

`helmet_head_edgevision_public_bike_augmented.pt` is an **experimental**
EfficientNet-B0 candidate; it is not selected in `configs/default.yaml`. It
continues from `helmet_head_edgevision.pt` with labeled examples from the CC
BY 4.0 [Bike Helmet Detection dataset](https://huggingface.co/datasets/shravya11/bike-helmet-dataset).
Although crop-level helmet recall and macro-F1 improved on a mixed test split,
the candidate falsely classified bare heads in the target close-up. This
public-bike candidate remains unselected; the headwear candidate documented below
was rolled back after the full traffic photo regressed; EdgeVision is active. See
`data/outputs/public-helmet-training/head-classifier-cpu/evaluation.json` and
`dataset/public-bike-helmet/ATTRIBUTION.md` for the comparison and attribution.

`data/outputs/kaggle-motorcycle-training/helmet-detector/weights/best.pt` is a
YOLO helmet-detector fine-tune using the Apache-2.0 Kaggle rider/helmet data
and the existing EdgeVision evaluation splits. It is **not** the configured
detector: on the held-out test split it scored mAP@50 0.612 versus 0.659 for
`safety_helmet_yolov8.pt`, and mAP@50:95 0.213 versus 0.353. It also added a
false helmet box on a scooter mirror in the supplied close-up. The training
data was small (83 Kaggle images), so this result does not establish that
motorcycle-specific training is unhelpful; it shows that this dataset and
fine-tune are not sufficient for a safe replacement.

`data/outputs/kaggle-motorcycle-training/hf-project/best.pt` is the Apache-2.0
checkpoint from [aneesarom/Helmet-Violation-Detection](https://huggingface.co/aneesarom/Helmet-Violation-Detection).
Its four classes include with-helmet and without-helmet, plus rider and plate.
The model card reports 120 annotated images; it produced a mirror false
positive on the target close-up and its detector pass took 7.1 seconds on this
CPU, so it is retained for comparison only.

`data/outputs/makeml-helmet-training/helmet-768-f25-e5-stable/weights/best.pt`
is an experimental three-epoch fine-tune on the public [MakeML/Kaggle helmet
dataset](https://www.kaggle.com/datasets/andrewmvd/helmet-detection) mixed
with EdgeVision. It did not beat the current detector on the held-out MakeML
test (mAP@50 0.828 vs. 0.847) and was effectively tied on EdgeVision (0.689 vs.
0.689). It is not configured for production. See
`data/outputs/makeml-helmet-training/evaluation.json` and
`dataset/kaggle-makeml-helmet/ATTRIBUTION.md`.

`data/outputs/roboflow-training/helmet-v6-edgevision-safe-e2/weights/best.pt`
is a two-epoch candidate fine-tuned on EdgeVision plus the filtered CC BY 4.0
[Roboflow helmet/no-helmet v6 dataset](https://universe.roboflow.com/gw-khadatkar-and-sv-wasule/helmet-and-no-helmet-rider-detection/dataset/6).
The export added only 95 new training images after duplicate and annotation
quality filtering. On the combined held-out test, candidate mAP@50/recall were
0.642/0.564 versus 0.655/0.587 for the active model; no-helmet recall declined
from 0.471 to 0.437. It is retained for reproducibility and is not active.
See `data/outputs/roboflow-training/REPORT.md` and
`dataset/roboflow-helmet-nohelmet-v6/prepared/ATTRIBUTION.md`.


`helmet_head_headwear_20261010.pt` is an experimental EfficientNet-B0 classifier (rolled back after full-scene regression), fine-tuned on the established EdgeVision heads plus filtered worn hats and unaugmented CS173 heads. On 180 held-out hat crops, helmet false positives at the app thresholds decreased from 47 to 8; helmet argmax recall stayed at 189/198. EdgeVision macro F1 decreased slightly (0.9204 to 0.9168). The cap and occluded-rider pipeline regressions pass. See `data/outputs/headwear-finetuning-2026-10-10/REPORT.md` for the publisher-label limitations, training settings and rollback. `helmet_head_edgevision.pt` remains the original checkpoint.

### YOLO26 Medium comparison (2026-10-10)

`yolo26m.pt` was downloaded from the official Ultralytics release and tested as a replacement for the Nano person/vehicle detector. It remains experimental: on two deduplicated reviewed images, Medium matched 60/71 motorcycles with 7 false detections, versus Nano's 68/71 with 1 false detection. The default configuration remains Nano. `configs/yolo26-medium-experimental.yaml` also tests matched head crops; that approach improves the example traffic photo but has not passed broader head-review validation. See `data/outputs/yolo26-medium-comparison/REPORT.md` for evidence, timing, and limits.
