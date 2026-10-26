# Changelog

All notable changes are listed here. The format follows Keep a Changelog, and
versions use semantic versioning.

## 1.0.1 - 2026-10-26

### Changed

- Replaced the documentation visual system with a monochrome halftone hand
  field, derived from the hand landmark data.
- Added Bodoni Moda and IBM Plex Mono for display and technical labels, with
  SIL Open Font License files.
- Rebuilt the README badges and system map in black, white, and gray.
- Updated the release archive to include the refreshed documentation assets.

### Fixed

- The README no longer uses an external status badge that introduced color into
  the monochrome header.
- The system map now labels the optional classifier and the training path
  without overlapping annotations.

There are no runtime behavior changes in this release.

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
