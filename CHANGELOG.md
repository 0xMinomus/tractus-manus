# Changelog

All notable changes are listed here. The format follows Keep a Changelog, and
versions use semantic versioning.

## 1.0.0 - 2026-10-26

First public release.

### Added

- Windows cursor control from a webcam with MediaPipe Hand Landmarker.
- One hand operation with automatic left hand takeover when the preferred hand
  leaves the frame.
- Touch-to-click on thumb and index fingertips.
- Near-pinch hold and drag, with a 300 ms activation window.
- Double click by touching twice.
- Right click, optional middle click, two finger scroll, and fist pause.
- One Euro cursor smoothing, active area mapping, and a click anchor.
- Live preview with landmarks, gesture state, progress indicators, and a screen
  map.
- Runtime keys for pause, smoothing, sensitivity, hand labels, debug, topmost,
  and middle click.
- Optional landmark collector and PyTorch trainer.
- Five model classifier ensemble with label smoothing and NumPy inference.
- Session-aware validation, chronological fallback validation, per-class
  precision, recall, and F1.
- MIT license, security policy, third-party notices, CI, and release archives.
