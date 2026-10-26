# Configuration

Tractus Manus reads `config.json` from the repository root. If the file is
missing, it uses the defaults in `handtrack/config.py`. To start from a
complete public example, copy `config.example.json` to `config.json`.

The application validates the file at startup. Unknown keys produce a warning.
Invalid values stop the application with an error that names the field.

```bat
copy config.example.json config.json
notepad config.json
```

Changes are read at startup. Runtime keys such as smoothing, sensitivity, and
middle click change the current session only. Edit the file for a permanent
change.

## Camera

| Key | Default | Meaning |
| --- | --- | --- |
| `camera.index` | `0` | Webcam index. Try `1` or `2` if another camera owns index 0. |
| `camera.width` | `1280` | Requested capture width. |
| `camera.height` | `720` | Requested capture height. |
| `camera.fps` | `60` | Requested capture rate. The camera may negotiate a lower rate. |
| `camera.backend` | `dshow` | Capture backend: `dshow`, `msmf`, or `any`. |
| `camera.mirror` | `true` | Show the preview as a mirror. This is the expected view for a selfie camera. |
| `camera.swap_handedness` | `true` | Swap the MediaPipe left and right labels when the labels are reversed. |

## Tracking

| Key | Default | Meaning |
| --- | --- | --- |
| `tracking.process_width` | `800` | Width used for MediaPipe. Lower values improve frame rate; higher values improve detail. `0` keeps the camera native size. |
| `tracking.max_hands` | `2` | Detect one or two hands. Only one hand controls the cursor at a time. |
| `tracking.min_detection_confidence` | `0.6` | Hand detection threshold. |
| `tracking.min_hand_presence_confidence` | `0.55` | Presence threshold for the hand landmarker. |
| `tracking.min_tracking_confidence` | `0.55` | Tracking threshold between frames. |

## Cursor

| Key | Default | Meaning |
| --- | --- | --- |
| `control.primary_hand` | `auto` | `auto` prefers the right hand, then the left. Use `right` or `left` to force one. |
| `control.target_monitor` | `primary` | `primary` maps to the primary monitor. `virtual` maps to the full virtual desktop. |
| `control.click_anchor_ms` | `60` | Cursor freeze after a touch click or hold starts. Higher values make the click more stable but delay drag. |
| `control.reacquire_ms` | `150` | Smooth glide time after a freeze. |
| `control.sensitivity` | `1.2` | Zoom factor for the active area. Higher values make screen corners easier to reach. |
| `control.active_area.x_min` | `0.12` | Left edge of the camera region mapped to the screen. |
| `control.active_area.x_max` | `0.88` | Right edge of the mapped region. |
| `control.active_area.y_min` | `0.1` | Top edge of the mapped region. |
| `control.active_area.y_max` | `0.8` | Bottom edge of the mapped region. |
| `control.smoothing.min_cutoff` | `2.5` | One Euro filter cutoff. Lower values are steadier and slower. |
| `control.smoothing.beta` | `7.0` | How quickly the filter reduces lag during fast motion. |
| `control.smoothing.derivative_cutoff` | `1.8` | Cutoff for the speed estimate used by the filter. |

The active area is applied before sensitivity. A smaller camera box maps the
same hand travel to a larger screen movement.

## Gestures

Pinch distances are normalized by the palm length. They are stable across
camera resolutions and hand sizes.

| Key | Default | Meaning |
| --- | --- | --- |
| `gestures.touch_press` | `0.18` | Distance at or below this value counts as fingertips touching. This sends a click. |
| `gestures.touch_release` | `0.26` | Distance above this value clears the touch latch. The next touch can click again. |
| `gestures.pinch_press` | `0.36` | Maximum distance for entering the near-pinch hold zone. |
| `gestures.pinch_release` | `0.52` | Distance that releases a held left button. |
| `gestures.pinch_margin` | `0.75` | The winning finger must be at most this fraction of the runner up distance. This prevents a right click from becoming a left click. |
| `gestures.pinch_stable_ms` | `25` | Time a pinch must remain stable before it is accepted. |
| `gestures.click_cooldown_ms` | `100` | Short delay after a click before a new pinch can start. |
| `gestures.left_hold_ms` | `300` | Time in the near-pinch zone before the left button is held for drag. |
| `gestures.middle_click_press` | `0.18` | Press distance for the tight thumb and ring finger pinch. |
| `gestures.middle_click_release` | `0.32` | Release distance for the middle click pinch. |
| `gestures.left_click` | `true` | Enable left click and drag. |
| `gestures.right_click` | `true` | Enable right click. |
| `gestures.middle_click` | `true` | Enable middle click. `M` or `Ctrl+M` toggles it while running. |

The relation must be:

```text
touch_press < touch_release < pinch_press
```

### Scroll

| Key | Default | Meaning |
| --- | --- | --- |
| `gestures.scroll.enabled` | `true` | Enable scroll mode. |
| `gestures.scroll.enter_hold_ms` | `120` | Time the two finger pose must be held before scroll starts. |
| `gestures.scroll.exit_hold_ms` | `150` | Time the pose may disappear before scroll ends. |
| `gestures.scroll.step_px` | `40` | Hand travel per wheel notch. |
| `gestures.scroll.smooth` | `0.8` | Scroll motion smoothing. `1.0` disables it. |
| `gestures.scroll.invert` | `false` | Reverse the wheel direction. |

### Pause

| Key | Default | Meaning |
| --- | --- | --- |
| `gestures.pause.enabled` | `true` | Enable pause with a closed fist. |
| `gestures.pause.hold_ms` | `1000` | Fist hold time needed to pause. |

## Optional classifier

The public release sets `enabled` to `false` and does not include a personal
model. The geometric engine works without it.

| Key | Default | Meaning |
| --- | --- | --- |
| `gestures.classifier.enabled` | `false` | Load the personal classifier model. |
| `gestures.classifier.model_path` | `models/gesture_classifier.npz` | Relative to the repository root, or an absolute path. |
| `gestures.classifier.confidence` | `0.65` | Minimum smoothed confidence before the model overrides gesture selection. |
| `gestures.classifier.smoothing` | `0.6` | Exponential smoothing of class probabilities. Higher is steadier and slower. |

See [Training](TRAINING.md) for the model workflow.

## Interface

| Key | Default | Meaning |
| --- | --- | --- |
| `ui.show_preview` | `true` | Show the camera preview and HUD. |
| `ui.always_on_top` | `true` | Keep the preview above other windows. Toggle with `T`. |
| `ui.show_skeleton` | `true` | Draw hand landmarks and connections. |
| `ui.debug` | `false` | Start with the debug panel visible. Toggle with `D`. |

## Complete example

`config.example.json` is the source of truth for a public install. The release
ZIP includes a copy named `config.json` with the optional classifier disabled.
