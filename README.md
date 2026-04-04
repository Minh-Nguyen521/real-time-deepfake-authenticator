# Real-Time Deepfake Authenticator

This repository contains a practical RLNet-style deepfake video baseline built around a `ResNet50 + LSTM` model and frame-sequence datasets such as `UADFV`.

The current project status is:

- training, single-sample prediction, visualization, and whole-dataset CSV export are implemented
- the pipeline works on extracted frame folders under `UADFV/.../frames` and `processed_data_keyframes/{real,fake}/...`
- transfer learning with pretrained `ResNet50` weights is enabled by default
- this is still a baseline system, not a paper-exact reproduction or a tuned production model

## Repository layout

```text
.
├── UADFV/
├── predict.py
├── test_dataset.py
├── train.py
├── visualize.py
└── rlnet/
    ├── __init__.py
    ├── data.py
    ├── metrics.py
    ├── model.py
    └── utils.py
```

## Dataset layout

The code expects this structure:

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

The model currently trains and evaluates from the extracted PNG frame folders, not by decoding raw videos on the fly.

It also supports this processed keyframe layout:

```text
processed_data_keyframes/
├── real/
│   └── <video_id>/
│       ├── frame0.jpg
│       ├── frame1.jpg
│       └── ...
└── fake/
    └── <video_id>/
        ├── frame0.jpg
        ├── frame1.jpg
        └── ...
```

## Current model

The baseline in [`rlnet/model.py`](./rlnet/model.py) uses:

- `ResNet50` backbone for per-frame spatial features
- `LSTM` temporal encoder
- bidirectional temporal modeling by default
- mean/max temporal pooling before the final classifier
- pretrained ImageNet `ResNet50` weights by default

## Data loading

The dataset loader in [`rlnet/data.py`](./rlnet/data.py) is lazy:

- frame directories are discovered up front
- frame images are opened only when a sample is requested
- frame path lists are cached after first access

This keeps memory use lower than preloading all extracted frames.

## Environment

Verified in this workspace with:

- Python `3.14`
- `torch 2.11.0`
- `torchvision 0.26.0`
- `tqdm 4.67.3`

## Train

Run training with:

```bash
python train.py --dataset-root UADFV --output-dir artifacts/rlnet_run
```

Important current defaults in [`train.py`](./train.py):

- `pretrained_backbone=True`
- `freeze_backbone=True`
- `unfreeze_epoch=3`
- `learning_rate=3e-4`
- `backbone_learning_rate=3e-5`
- `bidirectional=True`
- `epochs=10`

The training loop includes:

- stratified train/validation split
- `tqdm` progress bars
- separate learning rates for backbone and head
- validation-loss scheduler
- checkpoint saving by best validation F1

Example extended run:

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
  --backbone-learning-rate 3e-5
```

Disable transfer learning for comparison:

```bash
python train.py --no-pretrained-backbone --no-freeze-backbone
```

Quick smoke test:

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

- `best.pt`
- `history.json`
- `split.json`

The script prints the active training configuration at startup, so you can confirm pretrained weights are enabled.

Generate training and evaluation visual summaries later from saved artifacts:

```bash
python3 visualize_results.py \
  --checkpoint artifacts/rlnet_run/best.pt
```

If you already ran [`test_dataset.py`](./test_dataset.py), include its CSV to also render a dataset-level prediction summary:

```bash
python3 visualize_results.py \
  --checkpoint artifacts/rlnet_run/best.pt \
  --predictions-csv artifacts/dataset_predictions.csv
```

Default outputs go under `artifacts/rlnet_run/visualizations/`:

- `training_curves.png`
- `dataset_predictions_summary.png` when `--predictions-csv` is provided

## Predict one sample

Run inference on one extracted frame directory:

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

## Test whole dataset to CSV

Run the checkpoint across every discovered sample and export predictions:

```bash
python test_dataset.py \
  --checkpoint artifacts/rlnet_run/best.pt \
  --dataset-root UADFV \
  --output-csv artifacts/dataset_predictions.csv
```

Alternate processed-keyframe dataset:

```bash
python test_dataset.py \
  --checkpoint artifacts/rlnet_run/best.pt \
  --dataset-root processed_data_keyframes \
  --output-csv artifacts/processed_keyframes_predictions.csv
```

CSV columns:

- `video_id`
- `label`
- `label_name`
- `predicted_label`
- `prob_fake`
- `correct`
- `frame_dir`
- `video_path`

The script also prints dataset-level:

- accuracy
- precision
- recall
- F1

## Visualize predictions

Create a contact sheet, temporal probability plot, and JSON summary for one frame directory:

```bash
python visualize.py \
  --checkpoint artifacts/rlnet_run/best.pt \
  --frames-dir UADFV/real/frames/0000
```

Default outputs go under `artifacts/visualizations/<video_id>/`:

- `contact_sheet.png`
- `temporal_probabilities.png`
- `summary.json`

Example console output:

```text
frames_dir=UADFV/real/frames/0000
predicted_label=real
prob_fake=0.4828
contact_sheet=artifacts/visualizations/0000/contact_sheet.png
temporal_plot=artifacts/visualizations/0000/temporal_probabilities.png
summary=artifacts/visualizations/0000/summary.json
```

## Notes

- The pretrained `ResNet50` weights may load instantly without showing a download, because they are cached locally by `torchvision`.
- On this machine, the cached checkpoint file is under the Torch hub cache, so no visible re-download is expected once it is present.
- The dataset is small, so performance is very sensitive to the split and hyperparameters.
- Transfer learning is strongly recommended. Training from scratch can easily stay near chance.
- The current pipeline does not yet include face cropping, video decoding, k-fold evaluation, or a production inference app.

## Current limitations

- This is not an exact reproduction of the 2025 RLNet paper.
- The current dataset test script evaluates all discovered samples, but it does not yet export confusion matrices or threshold sweeps.
- The model is still under active tuning; low accuracy on a checkpoint usually means that checkpoint needs better training rather than that the scripts are broken.

## Next useful improvements

- add confusion matrix and threshold sweep export to `test_dataset.py`
- add k-fold cross-validation for more stable evaluation on `UADFV`
- add face-cropping preprocessing
- support direct video decoding at inference time

## Dataset Link
https://www.kaggle.com/datasets/maysuni/wild-deepfake
https://www.kaggle.com/datasets/adityakeshri9234/uadfv-dataset

## Core Ideas from this paper
https://www.researchgate.net/publication/393555522_Design_and_development_of_an_efficient_RLNet_prediction_model_for_deepfake_video_detection
