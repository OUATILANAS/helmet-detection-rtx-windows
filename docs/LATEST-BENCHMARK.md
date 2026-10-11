# Emertxe benchmark result — 2026-10-11

Completed CPU benchmark after load/label/geometry and tracker prechecks. Pinned source commit `630822b17d6283c05b772b1ef3e219539b90fb76`; original upstream modules are retained in training/vendor/emertxe-helmet-630822b. No training or production configuration changes were made for this benchmark.

| System / profile | Overall F1 | Video F1 | Still F1 |
|---|---:|---:|---:|
| Previous best RT-DETR + classifier | 0.634 | 0.563 | 0.662 |
| External stock detector | 0.688 | 0.578 | 0.744 |
| External best overall on VAL | 0.693 | 0.578 | 0.758 |
| External resolution 1280 | 0.655 | 0.644 | 0.656 |
| External tiled detector | 0.546 | 0.658 | 0.441 |

| Original video stage | Video F1 | Detected physical heads / 40 |
|---|---:|---:|
| tracked_raw | 0.500 | 8 |
| associated_current | 0.465 | 7 |
| original_temporal | 0.385 | 7 |

**Conclusion:** The external detector improves this small validation comparison, but the original full pipeline reduces video performance. Keep the model as an experimental candidate; do not replace production with this pipeline. Neither the best detector nor the full pipeline reaches 0.8 macro F1.

The best-overall detector has helmet F1 0.933 and no-helmet F1 0.453: averaging the two gives 0.693. It finds only 34 of 86 eligible physical heads. High helmet accuracy does not compensate for missing bare heads and non-riders. Higher resolution and tiles improve some video instants while harming still-image precision; report those profiles separately rather than combining their best columns.

The original ByteTrack path retains 8 of 40 eligible physical heads at the eight scored video instants; rider/motorcycle association reduces this to 7. The temporal decision makes no correct no-helmet decision at these instants, with two abstentions and the remaining cases missed. Upstream association allows one rider per motorcycle; passenger and non-rider coverage require separate verification.

## Scope and limits

- Same frozen 16 fully reviewed validation frames as the local three-model comparison: 8 video instants and 8 still images. Strict matching IoU 0.3, native head edge at least 8 pixels, missed heads and unmatched predictions counted. Threshold/resolution choices were made on validation only.
- Four continuous original video prefixes totaling 4,289 frames supplied tracking history; accuracy was scored only at the eight annotated instants. These are frame metrics, not a track-level violation benchmark. At scoring instants the exact frozen JPEG was used, after verifying correspondence with the decoded frame.
- Checkpoint metadata names the training run `ridesafe_yolov8s`, 100 epochs at 1280. Our validation video source is also RideSafe. External training identities/splits are unavailable here; overlap is unverified. Results cannot establish unseen-video generalization.
- The frozen 56-frame broad test queue remains untouched because annotation is only 29/56 complete and its policy prohibits candidate inference until all review is complete. Motorcycle-absent and non-rider performance are not certified by this benchmark.
- Static detector scores include every detected head. Complete-pipeline scores use its associated rider records and original temporal confidence rules. For strict head geometry, only actual current matched head boxes are scored; rider boxes or stale head boxes are never substituted.
- Inspected `final/src/temporal.py` has a three-observation minimum and 0.7 confidence threshold. It does not implement the 0.67 vote-agreement condition advertised in the repository README. The benchmark uses the actual pinned code.

## Artifacts

- `PLAN.json`: frozen inputs, source hashes and evaluation settings.
- `PREFLIGHT.json`: CPU load, memory, timing, class mapping and tracker checks.
- `RESULTS.json`: complete metrics and saved local references.
- `STATIC-PREDICTIONS.json` and `TEMPORAL-PREDICTIONS.json`: reproducible head outputs.
- `STATIC-REVIEW.jpg`: inspected overlays showing distant misses, duplicate class predictions and a missed cap-wearing non-rider.
- `MEMBERSHIP-DIAGNOSTIC.json`: reviewed video role confusion with misses retained.

Next step: finish broad test annotations, then test a fixed external detector profile alongside the retained local candidate. If it generalizes, evaluate it inside our existing association/status pipeline before any deployment. Do not fine-tune using the user test pictures or tune settings on the broad test.
