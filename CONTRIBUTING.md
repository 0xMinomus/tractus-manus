# Contributing

Thank you for helping improve Tractus Manus. The project is small and keeps
the gesture engine testable without a webcam.

## Setup

1. Install [uv](https://docs.astral.sh/uv/).
2. From the repository root, run:

```bat
uv python install 3.12
uv sync --group dev
copy config.example.json config.json
```

3. Run the tests and the linter:

```bat
uv run pytest -q
uv run ruff check .
```

The unit tests use synthetic landmarks. They do not open a camera. The trainer
test skips when PyTorch is not installed. To run it, install the optional
training dependency:

```bat
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu128
```

## What to change

The gesture behavior is defined by small, testable pieces:

- `handtrack/features` live in `handtrack/hands.py` and `handtrack/classifier.py`.
- Thresholds and state transitions live in `handtrack/gestures.py`.
- Screen mapping and cursor filtering live in `handtrack/cursor.py` and `handtrack/mapping.py`.
- Windows input lives in `handtrack/input.py`.

When you change a threshold, add a test with synthetic landmarks in
`tests/test_gestures.py` or `tests/test_classifier.py`. When you change a
config field, update `handtrack/config.py`, `config.example.json`, and
`docs/CONFIGURATION.md`.

## Pull requests

- Keep one behavior change per pull request.
- Run `uv run pytest -q` and `uv run ruff check .` before opening the request.
- Describe the user visible effect and the test that covers it.
- Do not commit recordings, personal classifier weights, `.venv`, or local
  `config.json` files. See `.gitignore`.
- Do not include webcam captures or face images in issues or pull requests.

## Writing style

Documentation is written in English. The preview HUD and some console messages
are in Indonesian. Keep user facing strings clear and specific. Avoid em dashes
and marketing claims without evidence.

## Security

Do not report a vulnerability in a public issue. See [SECURITY.md](SECURITY.md).
