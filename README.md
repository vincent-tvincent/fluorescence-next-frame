# Fluorescence next-frame prediction

Train one recurrent U-Net to predict the next raw fluorescence image in a time-lapse movie. Each training example consists of consecutive observations and their next recorded frame. The recorded intensities supply all supervision.

## Install

Use Python 3.10 or newer. From the repository root, run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
fluoroframe --help
```

## Input

Point each movie argument at a directory containing one grayscale TIFF per time point. A single plane has shape `Y X` and a volume has shape `Z Y X`. Keep the frame shape, channel, and time interval consistent within a movie. Use a separate directory per position and channel.

```text
data/
  Pos0/SPIMA/SPIMA_t001.tif
  Pos0/SPIMA/SPIMA_t002.tif
  ...
  Pos1/SPIMA/SPIMA_t001.tif
  Pos1/SPIMA/SPIMA_t002.tif
  ...
```

Frame filenames share a prefix and end in a consecutive numeric index before `.tif`, `.tiff`, or `.ome.tif`. Discovery sorts by that number, skips hidden files and subdirectories, and reports gaps. For a missing time point, pass the contiguous runs as separate movies. For a single TIFF containing an entire movie, first export one grayscale TIFF per time point. Compressed TIFF codecs such as LZW may require `python -m pip install imagecodecs`.

## Train

```bash
fluoroframe train --train data/Pos0/SPIMA --val data/Pos1/SPIMA --run runs/first
```

With one movie, omit `--val` to reserve the last 20% of frames for validation. With multiple training directories and no explicit validation set, the final directory is held out. Splitting happens before clip construction, and each segment needs at least two frames. Use an independently acquired movie for the strongest generalization check.

The encoder has 16, 32, and 64 channels by default. A convolutional GRU updates a spatial hidden state once per observed frame. The decoder combines that state with skip connections from the most recent frame and predicts a signed intensity change. Adding the change to the observed frame gives the next image. `--history 1` at inference corresponds to `n = 0` in the interval `t-n ... t`.

Training randomly samples 1, 2, 4, or 8 observations, limited to histories the movies can provide. Each optimizer update uses one clip by default. Set `--batch-size N` for a minibatch of independently sampled clips of the same length and crop shape. Within a clip, the observations and target share crop coordinates. The loss averages the absolute error over the target voxels and clips. The default crop is `96 96` in 2D and `12 96 96` in 3D. One offset and scale are estimated from sampled training voxels and reused for validation and prediction.

For a small 3D trial, run

```bash
fluoroframe train --train data/Pos0/SPIMA --run runs/trial \
  --histories 1 2 4 --patch 8 48 48 --width 8 --epochs 2 --steps-per-epoch 8
```

Each run creates a new directory. `run.json` records settings and split boundaries. `metrics.jsonl` reports MAE in the original intensity units. Validation compares each available history and copying the last observation on the same fixed target crops. `best.pt` is the epoch with the lowest validation MAE for the longest evaluated history. Device selection tries CUDA, then Apple MPS, then CPU. Use `--device cpu` to select CPU explicitly.

### Unattended FP32 training on the Pavak movies

Run the following from the checkout, substituting the mounted data and output paths. The output directory must be new. Set `PYTHON` to the interpreter in your PyTorch environment if needed.

```bash
bash scripts/train_pavak.sh /mnt/hdd/2026_bright_channel /mnt/hdd/forecast-runs/pavak-01
```

The script reads the existing per-frame TIFFs in place and writes the entire run under the supplied output directory. It requests 200 epochs of 64 optimizer updates, uses Pos0–Pos6 for training, and selects `training/best.pt` by eight-frame validation MAE on Pos7. It also saves `training/last.pt` after every completed epoch. After training, it scores the best checkpoint on up to 100 Pos8 crops and writes `pos8.json`. Training progress appears in the terminal and `console.log`.

CUDA training uses FP32 tensors and IEEE FP32 convolutions. With `--batch-size auto`, each disposable trial runs two full training updates at the longest history and largest crop. The probe doubles the batch size until a trial fails, then tests integer sizes inside the bracket. It selects the largest passing batch while reserving 1 GiB of free device memory by default. A fresh model and AdamW optimizer start the real run. `training/run.json` records the selected size and probe results. The available headroom depends on the GPU's other workloads.

## Predict

To predict frame 14 using frames 10 through 13 in a held-out movie, run

```bash
fluoroframe predict --checkpoint runs/first/best.pt --movie data/Pos2/SPIMA \
  --through SPIMA_t013.tif --history 4 --output runs/predicted_t014.tif
```

Use `--history 1` for the latest image alone. `--history all` streams from the beginning of the movie through the chosen observation. Omitting `--through` forecasts the time point immediately after the last file. The hidden state starts fresh for each movie and its size is independent of the observation count.

The output is a full-size float32 TIFF in the original intensity units. Inference reads each observed image once and processes spatial tiles with overlapping context. Only the center of each tile is written. The default tile core matches the training crop. If needed, use `--tile Z Y X` or `--tile Y X` to change it. A tile as large as the frame evaluates the whole field at once. Use a new output filename outside the input movie directory.

## Evaluate

```bash
fluoroframe evaluate --checkpoint runs/first/best.pt --movies data/Pos2/SPIMA \
  --histories 1 4 --samples 32
```

`copy_mae` is the error of using the last observed frame as the forecast. `model_mae` gives the error for each requested observation count. Lower is better. Every method sees the same target times and spatial crops. The report includes the number of crops and scored voxels. To score whole images, set `--patch` to the frame shape and choose a device with sufficient memory.

Run the CPU smoke tests, including a training and prediction round trip on synthetic data, with `python -m unittest discover -s tests`.
