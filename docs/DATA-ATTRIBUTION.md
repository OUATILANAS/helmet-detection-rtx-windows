# EdgeVision helmet fine-tuning data

Source: [EdgeVision Dataset, Mendeley Data V1](https://data.mendeley.com/datasets/j82bnw7gsr/1), DOI [10.17632/j82bnw7gsr.1](https://doi.org/10.17632/j82bnw7gsr.1).

The dataset contains 2,392 motorcycle traffic images and 8,275 annotations.
It is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Recommended attribution: Gajjar, Kandarp; Patel, Nancy; Patel, Ridham; Patel,
Krutika; Dabhi, Chrisha; Trivedi, Himani; Goyani, Mahesh; Bhatt, Chintan
(2026), “EdgeVision Dataset”, Mendeley Data, V1.

The original images and YOLO labels are under `source/`. The prepared training
set is under `yolo/`; `split_manifest.json` records the file split and annotation
counts. Preparation drops the broad BikeWithRider boxes, maps Helmet to class 0
and NoHelmet to class 1, and groups perceptual near-duplicates (64-bit pHash
distance at most 4) before splitting. The image files in the prepared set are
symlinks to the originals, so they do not consume a second copy of the image
data. Labels and split files are modified from the source dataset.

The dataset has a smartphone-captured road-scene distribution. Treat its held
out metrics as domain-specific evidence; inspect the target CCTV and user
photos separately before replacing the app checkpoint.
