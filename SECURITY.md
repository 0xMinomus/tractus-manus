# Security policy

## Supported versions

Security fixes are provided for the 1.x release line. The current release is
listed on the [releases page](https://github.com/0xMinomus/tractus-manus/releases).

## Reporting a vulnerability

Please use GitHub private vulnerability reporting:

https://github.com/0xMinomus/tractus-manus/security/advisories/new

Do not open a public issue for a vulnerability. Include the affected version,
the Windows version, the steps to reproduce, and the impact. Do not attach
webcam images, landmark recordings, or other personal data. If a recording is
necessary to reproduce a bug, describe the gesture and use synthetic
landmarks instead.

There is no bug bounty program. Reports are acknowledged as time allows.

## Scope and data flow

Tractus Manus is a local Windows desktop application.

- Camera frames are processed in memory by OpenCV and MediaPipe. The runtime
  does not write frames, images, video, or audio to disk.
- The optional data collector writes one JSON line per saved frame. Each line
  contains 21 hand landmark coordinates, a label, a session name, the frame
  width and height, and a timestamp. It contains no pixels and no face data.
- The training data directory, backups, and personal classifier model are
  excluded from Git and from release ZIP files.
- There is no telemetry, account system, analytics endpoint, or cloud storage.
- The runtime downloads the MediaPipe hand landmark model only if
  `models/hand_landmarker.task` is missing. The download is checked against a
  pinned SHA-256 hash. Release archives include the model.
- Model files are loaded with NumPy `allow_pickle=False`.
- The app calls Windows `SendInput` to move the cursor and press mouse buttons.
  It does not install a driver or a service.
- Running as administrator is optional. Elevated applications may reject
  input from a non-elevated process.

## Dependencies

Third-party licenses and sources are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Install release archives only
from this repository's releases page. Verify the published SHA-256 checksum
before running a downloaded archive.

## Out of scope

- Applications that ignore synthetic input, including some exclusive
  fullscreen games and anti-cheat protected applications.
- Camera, driver, or Windows permission problems.
- Physical access attacks and malware already running on the machine.
