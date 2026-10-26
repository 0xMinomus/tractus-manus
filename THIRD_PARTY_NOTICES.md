# Third-party notices

Tractus Manus is MIT licensed. It uses the following third-party components.
Their licenses apply to those components.

## MediaPipe Hand Landmarker model

- Component: `models/hand_landmarker.task`
- Source: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
- License: Apache License 2.0
- Documentation: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

The model is included in release archives. It is not trained by this project.
A copy of the Apache License 2.0 is included at `licenses/Apache-2.0.txt`.

## MediaPipe Python package

- Package: `mediapipe`
- License: Apache License 2.0
- Source: https://github.com/google-ai-edge/mediapipe

## OpenCV

- Package: `opencv-contrib-python`
- License: Apache License 2.0
- Source: https://github.com/opencv/opencv

## NumPy

- Package: `numpy`
- License: BSD 3-Clause
- Source: https://github.com/numpy/numpy

## PyTorch, optional

- Package: `torch`
- License: BSD 3-Clause
- Source: https://github.com/pytorch/pytorch

PyTorch is only needed for training a personal classifier. It is not part of
the default runtime installation.

## uv

- Tool: `uv`
- License: MIT or Apache 2.0
- Source: https://github.com/astral-sh/uv

## Fonts in documentation

The SVG diagrams use generic system sans-serif font families. No font files
are distributed with this repository.

## Trademarks

MediaPipe and Google are trademarks of Google LLC. Tractus Manus is not
affiliated with or endorsed by Google.
