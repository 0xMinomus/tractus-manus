# Training a personal classifier

The geometric gestures work without a classifier. A personal classifier helps
when pinch distances or finger poses are ambiguous. It selects a gesture label
for the gesture engine. It does not move the cursor.

This guide describes the current workflow. Training data and personal model
files stay on your machine and are excluded from Git and release archives.

## What is recorded

The collector saves hand landmarks, not camera images. Each JSON line contains:

- `session`: a session name.
- `label`: one of `neutral`, `left`, `right`, `middle`, `fist`, `scroll`.
- `hand`: `Left` or `Right`.
- `landmarks`: 21 rows of normalized `x`, `y`, `z`.
- `width` and `height`: the processed frame size.
- `t`: a timestamp.

There are no pixels, images, video, face data, or audio in the recording.

## Before recording

Close `run.bat` so the camera is free. If you want to keep an older dataset,
back it up:

```bat
copy data\gestures.jsonl data\backups\gestures_prev.jsonl
copy models\gesture_classifier.npz data\backups\classifier_prev.npz
```

To start a clean dataset, rename the current file. The collector creates a new
one:

```bat
ren data\gestures.jsonl gestures_prev.jsonl
```

## Record the data

```bat
collect.bat
```

The default target is 600 frames per label per session. The collector stops
automatically when a label reaches its target.

For `left`, the target is split into two blocks:

1. Press `2` and record the fingertips **touching** for 300 frames.
2. Press `Space` again and record the **near pinch** with a small gap for 300
   frames.

The HUD shows `blok 1/2` and `blok 2/2`. Do not create a new label for the
second block.

Record all six labels in the same session. The HUD shows `sesi x/6 label`.
Press `N` only after all six labels are complete. The collector refuses `N`
until then, so an incomplete session cannot be created by accident.

Aim for at least three complete sessions. Five sessions give a better estimate
of real world stability. Each session should contain all six labels:

| Label | Pose |
| --- | --- |
| `neutral` | Relaxed hand or pointing hand. Not a pinch, fist, or V sign. |
| `left` | Index and thumb, half touching and half near pinch. |
| `right` | Thumb and middle finger pinch, index finger away. |
| `middle` | Tight thumb and ring finger pinch. |
| `fist` | Closed fist. |
| `scroll` | Index and middle extended, ring and little finger folded. |

Vary distance from the camera, hand position in the frame, angle, and lighting
between sessions. Keep the same hand label setting for the whole dataset.
Changing `camera.swap_handedness` in the middle of a recording will make the
labels inconsistent.

## Train

```bat
train.bat
```

The trainer uses CUDA when PyTorch and a supported GPU are available. It uses
the CPU otherwise.

Defaults:

- five MLP models with different random seeds
- label smoothing `0.05`
- early stopping on a validation split
- class balancing for unequal sample counts

The validation split is session-aware. If every class appears in at least two
sessions, the trainer keeps sessions separate. If each session contains only
one label but each class has at least two sessions, it holds out one session
per class. If neither is possible, it uses the last part of each class block
and reports that the result is optimistic.

The output includes per-class support, recall, precision, and F1. Check the
classes with low recall. Add a session that includes those poses rather than
adding more frames to the same recording.

Options:

```bat
train.bat --ensemble 5 --label-smoothing 0.05
train.bat --ensemble 1
train.bat --label-smoothing 0
train.bat --data data\gestures.jsonl --out models\gesture_classifier.npz
```

The model is written to `models/gesture_classifier.npz`. It is ignored by Git.
The release archive does not include it.

## Enable the model

Edit `config.json`:

```json
{
  "gestures": {
    "classifier": {
      "enabled": true,
      "model_path": "models/gesture_classifier.npz",
      "confidence": 0.65,
      "smoothing": 0.6
    }
  }
}
```

Restart `run.bat`. The banner shows `AI gestur: aktif`. The debug overlay shows
the model label and confidence.

If the model is missing or unreadable, the application logs a warning and
continues with geometric rules.

## PyTorch installation

The runtime setup does not install PyTorch. On an NVIDIA GPU:

```bat
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

For a CPU only environment:

```bat
uv sync --group train
```

PyTorch is only used for training. Runtime inference uses NumPy.

## Tuning

- If the model is too slow to change labels, lower `classifier.smoothing` to
  `0.5` or `0.6`.
- If labels flicker, raise `classifier.smoothing` to `0.7` or `0.75`.
- If the model overrides correct geometric gestures too often, raise
  `classifier.confidence` to `0.7` or `0.8`.
- If the model is ignored, lower `classifier.confidence` to `0.6`.

Do not tune thresholds to compensate for inconsistent training poses. Record
the pose cleanly first.

## Validation limits

A frame holdout from one continuous recording can look close to perfect because
neighboring frames are almost identical. Treat a 100% frame score as a sanity
check, not proof of cross-session accuracy. The most useful number is the
per-class result on sessions that were not used for training.

## Data retention

`data/`, `data/backups/`, and `models/gesture_classifier.npz` are ignored by
Git. The release builder rejects training data, images, video, and personal
classifier files. Delete the data directory when you no longer need it.
