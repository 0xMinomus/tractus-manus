# Tractus Manus 1.0.0

Tractus Manus turns a Windows webcam into a mouse. This is the first public
release.

## Download

Download `tractus-manus-1.0.0-windows.zip`, extract it, and run `setup.bat`
once. Then run `run.bat`. The release includes the MediaPipe hand landmark
model.

## Highlights

- Cursor control from one or two hands, with right hand priority.
- Touch thumb and index fingertips to click.
- Hold a near pinch to drag.
- Touch twice for double click.
- Right click, optional middle click, scroll, and fist pause.
- One Euro cursor smoothing and a stable click anchor.
- Live preview with landmarks, state, and screen map.
- Optional personal classifier with a five model ensemble and label smoothing.
- Local processing. No camera images are saved or uploaded.

## Requirements

- Windows 10 or Windows 11, 64 bit.
- A webcam.
- uv for the one time setup.
- Python 3.12 is installed by setup.

## Privacy

The runtime processes frames in memory. The optional collector stores only hand
landmark coordinates and labels. This release does not include training
recordings, face images, or a personal classifier model.

## Known limitations

- The preview HUD and console messages are in Indonesian.
- Exclusive fullscreen and raw input games may ignore synthetic mouse input.
- The public configuration uses geometric gestures. Train a personal
  classifier for ambiguous pinches.

## Checksum

Verify the downloaded ZIP against the `.sha256` file attached to the release.
