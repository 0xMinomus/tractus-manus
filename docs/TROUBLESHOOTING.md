# Troubleshooting

## The camera does not open

Close applications that may own the camera: browsers, Teams, Zoom, OBS, and
other camera tools. Check Windows privacy settings under Settings, Privacy and
security, Camera. If index 0 belongs to another camera, try:

```bat
run.bat --camera 1
```

If DirectShow fails, try the Media Foundation backend in `config.json`:

```json
"camera": { "backend": "msmf" }
```

## The preview is black

A black frame is usually a privacy shutter, a camera permission issue, or a
camera driver that has not started. The application logs a warning after a few
seconds of black frames. Check the hardware shutter and the Windows camera
permission. Try another camera index.

## The camera runs but tracking is slow

- Lower `tracking.process_width` from `800` to `640` or `720`.
- Close other camera and video applications.
- Check that the camera is not delivering a very low frame rate. The log prints
  the negotiated rate.
- On a laptop, use the wall charger. Some GPUs and cameras reduce clocks on
  battery.

## The cursor is shaky

- Raise `control.smoothing.min_cutoff` slightly, for example from `2.5` to
  `3.0`. Higher values reduce lag but let more jitter through.
- Lower `control.sensitivity` if small hand movements move the cursor too far.
- Make sure the hand is well lit and not at the edge of the frame.
- Use `D` to inspect the landmarks and pinch distances.

## The cursor is too slow or feels laggy

- Raise `control.smoothing.min_cutoff` from `2.5` to `3.0` or `3.5` to reduce
  filter lag. Lower values smooth more but add lag.
- Raise `tracking.process_width` only if the machine has headroom. Lowering it
  usually improves frame rate.
- Check the preview FPS. A low camera FPS, not the classifier, is the usual
  cause of cursor lag.
- In a fullscreen application, press `T` to turn off always-on-top for the
  preview, or use `run.bat --no-preview`.

## A normal click never happens

- Check the debug overlay with `D`. The index pinch distance must fall to
  `gestures.touch_press` or below, default `0.18`.
- Touch the fingertips more firmly, or raise `touch_press` to `0.20` or `0.22`.
- `touch_release` must stay above `touch_press` and below `pinch_press`.
- After a click, open the pinch past `touch_release` before the next touch.
- The cursor is intentionally frozen during the click decision. This is not a
  freeze bug.

## A click becomes a drag

The fingers stopped in the near-pinch zone for longer than `left_hold_ms`,
default 300 ms. To click, touch the fingertips together. To drag, keep a small
gap and wait for the hold to engage, then move. Lower `left_hold_ms` to `250`
if you want a shorter hold delay.

## Double click is unreliable

Use two clean touch events. Open the pinch past `touch_release` between them.
The cursor is held in place briefly after a click so the second click lands at
the same point. Lower `click_cooldown_ms` to `70` or `80` if the second touch is
delayed.

## Right click is difficult

Keep the index finger away from the thumb when you pinch with the middle
finger. The engine rejects a pinch when two fingers are almost equally close.
The debug overlay shows `JEPIT AMBIGU` when this happens. You can lower
`pinch_margin` to `0.70` for a stricter choice, or raise it if valid clicks are
being rejected.

## Middle click happens accidentally

Press `M` or `Ctrl+M` to toggle middle click off for the session. For a permanent
change, set:

```json
"gestures": { "middle_click": false }
```

## Scroll starts by accident

Place the index and middle fingers clearly and fold the ring and little finger.
Raise `gestures.scroll.enter_hold_ms` from `120` to `180` if the mode starts too
easily. Lower it if scroll is hard to enter.

## Pause happens during normal movement

A closed fist held for about one second triggers pause. Avoid resting the hand
in a fist. Raise `gestures.pause.hold_ms` if needed, or disable it:

```json
"gestures": { "pause": { "enabled": false } }
```

## The preview is in front of a fullscreen app

Press `T` to toggle always-on-top. For a fullscreen app, run without a preview:

```bat
run.bat --no-preview
```

Some exclusive fullscreen applications and games with raw input ignore
synthetic input. A borderless window usually works better.

## Nothing happens in an elevated application

Windows blocks input from a normal process to an elevated window. Run Tractus
Manus as administrator only if you trust the target application. The app does
not require elevation for normal desktop use.

## The model is not loaded

Check the path and the file name:

```json
"gestures": {
  "classifier": {
    "enabled": true,
    "model_path": "models/gesture_classifier.npz"
  }
}
```

The banner should show the model status. A missing model is not fatal. The
geometric rules continue to work.

## Training says PyTorch is missing

Install the CUDA build for an NVIDIA GPU:

```bat
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu128
```

Or install the CPU training group:

```bat
uv sync --group train
```

## The model performs badly in daily use

Check per-class recall from the trainer. Record sessions with the hand at
different distances, positions, and angles. A frame holdout within one
recording can look much better than a new session. Use at least three complete
sessions for a realistic check.

## Training data contains no images

This is intentional. The collector stores hand landmark coordinates only. It
does not save camera frames. See [Training](TRAINING.md) and
[SECURITY.md](../SECURITY.md).

## I need to reset everything

Keep a backup first. To start a new dataset:

```bat
copy data\gestures.jsonl data\backups\gestures_prev.jsonl
ren data\gestures.jsonl gestures_prev.jsonl
```

To remove a personal model, delete `models/gesture_classifier.npz` and set
`gestures.classifier.enabled` to `false`.
