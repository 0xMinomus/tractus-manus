# Releasing

The release is a Windows ZIP for end users and a tagged GitHub release.

## Versioning

The version lives in these files and must match:

- `pyproject.toml`
- `handtrack/__init__.py`
- `handtrack/cli.py`

Use semantic versioning. Tags use a `v` prefix, for example `v1.0.1`.

## Before building

1. Run the tests and linter on Windows:

```bat
uv sync --group dev
uv run pytest -q
uv run ruff check .
```

2. Confirm that `config.json` is not tracked. It is a local file and may contain
   personal paths or a personal classifier setting.

3. Confirm that the release contains no training data, personal classifier, or
   camera media:

```bat
git ls-files | findstr /I "data/ gesture_classifier .png .jpg .jpeg .webp .mp4 .mov"
```

The command should print nothing except source assets. The repository does not
store training recordings or face images.

4. Review `CHANGELOG.md` and `docs/RELEASE_NOTES_v<version>.md`.

## Build the ZIP

```bat
.venv\Scripts\python.exe scripts\build_release.py
```

The builder creates:

- `dist/tractus-manus-1.0.1-windows.zip`
- `dist/tractus-manus-1.0.1-windows.zip.sha256`

The archive contains the application, the MediaPipe model, the public
`config.json` generated from `config.example.json`, tests, documentation, and
the launchers. It rejects datasets, personal classifier weights, and camera
media.

## Verify the archive

```bat
.venv\Scripts\python.exe -c "import zipfile; z=zipfile.ZipFile('dist/tractus-manus-1.0.1-windows.zip'); print('\n'.join(z.namelist()))"
certutil -hashfile dist\tractus-manus-1.0.1-windows.zip SHA256
```

Extract the ZIP to a clean folder and run `setup.bat` and `run.bat`. The
classifier is disabled in the public config, so the geometric gestures should
work without a personal model.

## Publish

```bat
git add .
git commit -m "release: v1.0.1"
git tag v1.0.1
git push origin main
git push origin v1.0.1
gh release create v1.0.1 dist\tractus-manus-1.0.1-windows.zip dist\tractus-manus-1.0.1-windows.zip.sha256 --title "Tractus Manus 1.0.1" --notes-file docs\RELEASE_NOTES_v1.0.1.md
```

The README links to the latest release:

https://github.com/0xMinomus/tractus-manus/releases/latest

## After publishing

- Confirm the CI badge passes on the tagged commit.
- Download the attached ZIP from the release page and verify the checksum.
- Do not edit a published release archive. Publish a new patch version instead.
