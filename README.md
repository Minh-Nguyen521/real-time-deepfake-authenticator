# Real-Time Deepfake Authenticator

This repository contains a practical baseline reimplementation of an RLNet-style deepfake video detector built around a `ResNet50 + LSTM` architecture.

The current workspace is centered on the `UADFV` dataset and uses pre-extracted frame sequences for training and inference. The implementation is inspired by the 2025 RLNet paper, but it is not a strict reproduction of the published experiments.

## What is included

- A frame-sequence dataset loader for `UADFV`
- An `RLNet` model with:
  - `ResNet50` as the spatial encoder
  - `LSTM` as the temporal sequence model
- A training script with:
  - stratified train/validation splitting
  - checkpoint saving
  - basic classification metrics
- A prediction script for running inference on a single frame directory
- A visualization script for annotated sampled frames and temporal probability plots
- Lazy frame loading with cached per-directory frame indexes

## Repository layout

```text
.
├── UADFV/
├── predict.py
├── test_dataset.py
├── visualize.py
├── train.py
└── rlnet/
    ├── __init__.py
    ├── data.py
    ├── metrics.py
    ├── model.py
    └── utils.py
```

## Dataset layout

The code expects the dataset to look like this:

```text
UADFV/
├── real/
│   ├── 0000.mp4
│   └── frames/
│       └── 0000/
│           ├── 000.png
│           ├── 001.png
│           └── ...
└── fake/
    ├── 0000_fake.mp4
    └── frames/
        └── 0000_fake/
            ├── 000.png
            ├── 001.png
            └── ...
```

The training and prediction pipeline currently reads from the extracted PNG frame folders under `real/frames` and `fake/frames`.

## Loading behavior

The dataset loader is lazy:

- it discovers video/frame directories up front
- it opens frame images only when a sample is requested
- it caches each frame directory listing after first access so later epochs avoid repeated directory scans

This keeps memory usage lower than preloading all extracted frames into RAM.

## Environment

The implementation was verified with:

- Python `3.14`
- `torch 2.11.0`
- `torchvision 0.26.0`
- `tqdm 4.67.3`

## Train

Run a standard training job:

```bash
python train.py --dataset-root UADFV --output-dir artifacts/rlnet_run
```

The default training recipe now uses:

- pretrained ImageNet `ResNet50` weights
- a frozen-backbone warm start, then backbone unfreezing
- a lower learning rate for the backbone than for the temporal/classification head
- `tqdm` progress bars for both training and validation batches

Useful options:

```bash
python train.py \
  --dataset-root UADFV \
  --output-dir artifacts/rlnet_run \
  --epochs 12 \
  --batch-size 2 \
  --sequence-length 16 \
  --image-size 224 \
  --hidden-size 256 \
  --num-layers 2 \
  --dropout 0.3 \
  --learning-rate 3e-4 \
  --backbone-learning-rate 3e-5 \
  --weight-decay 1e-4
```

If you want to disable transfer learning for comparison, use:

```bash
python train.py --no-pretrained-backbone --no-freeze-backbone
```

If you want to experiment quickly on a tiny subset:

```bash
python train.py \
  --dataset-root UADFV \
  --output-dir artifacts/rlnet_smoke \
  --epochs 1 \
  --batch-size 1 \
  --sequence-length 2 \
  --image-size 64 \
  --hidden-size 32 \
  --num-layers 1 \
  --max-train-samples 2 \
  --max-val-samples 2
```

Training outputs:

- `best.pt`: best checkpoint by validation F1
- `history.json`: per-epoch metrics
- `split.json`: train/validation video IDs

At startup, the script prints the active training configuration so you can verify that the pretrained backbone is enabled.

## Predict

Run inference on a single extracted frame directory:

```bash
python predict.py \
  --checkpoint artifacts/rlnet_run/best.pt \
  --frames-dir UADFV/real/frames/0000
```

Example output:

```text
video_id=0000
prob_fake=0.4828
predicted_label=real
```

## Test Whole Dataset

Run a checkpoint over every discovered sample and export the predictions to CSV:

```bash
./venv/bin/python test_dataset.py \
  --checkpoint artifacts/rlnet_run/best.pt \
  --dataset-root UADFV \
  --output-csv artifacts/dataset_predictions.csv
```

The CSV includes:

- `video_id`
- `label`
- `label_name`
- `predicted_label`
- `prob_fake`
- `correct`
- `frame_dir`
- `video_path`

The script also prints dataset-level accuracy, precision, recall, and F1 at the end.

## Visualize

Generate a contact sheet and a simple temporal probability plot for one frame directory:

```bash
python visualize.py \
  --checkpoint artifacts/rlnet_run/best.pt \
  --frames-dir UADFV/real/frames/0000
```

Example output:

```text
frames_dir=UADFV/real/frames/0000
predicted_label=real
prob_fake=0.4828
contact_sheet=artifacts/visualizations/0000/contact_sheet.png
temporal_plot=artifacts/visualizations/0000/temporal_probabilities.png
summary=artifacts/visualizations/0000/summary.json
```

What it saves:

- `contact_sheet.png`: sampled frames with per-step prefix probability and final video-level probability
- `temporal_probabilities.png`: line plot of the fake probability as more sampled frames are revealed
- `summary.json`: machine-readable prediction summary and sampled frame list

## Notes

- This is a practical baseline, not a paper-exact reproduction.
- The implementation uses pre-extracted frames instead of decoding videos on the fly.
- The paper reports a `ResNet50 + LSTM` configuration, but some experimental details are not fully specified in a reproducible way.
- The `UADFV` dataset in this repo is small, so results can vary significantly depending on the split and hyperparameters.
- Because the dataset is small, transfer learning is strongly recommended; training `ResNet50` from scratch will often hover near chance.

## Next improvements

Possible next steps:

- add face-cropping preprocessing
- support direct video decoding at inference time
- add confusion matrix and richer evaluation reports
- build a webcam or upload-based demo app
