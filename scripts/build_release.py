"""Build a reproducible Windows release ZIP without personal data."""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"

# Source files and documentation that belong in an end user archive.
INCLUDE_FILES = (
    ".python-version",
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "LICENSE",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "main.py",
    "collect_hand_data.py",
    "train_gesture.py",
    "collect.bat",
    "run.bat",
    "setup.bat",
    "config.example.json",
    "models/hand_landmarker.task",
    "models/README.md",
)
INCLUDE_DIRS = ("handtrack", "docs", "tests", "licenses")

FORBIDDEN_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "data",
    "backups",
}
FORBIDDEN_SUFFIXES = {
    ".jsonl",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".wav",
    ".mp3",
}
FORBIDDEN_NAMES = {"config.json", "gesture_classifier.npz"}


def version() -> str:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def collect() -> list[Path]:
    files: list[Path] = []
    for name in INCLUDE_FILES:
        path = ROOT / name
        if not path.is_file():
            raise SystemExit(f"Missing release file: {name}")
        files.append(path)
    for dirname in INCLUDE_DIRS:
        base = ROOT / dirname
        if not base.is_dir():
            raise SystemExit(f"Missing release directory: {dirname}")
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix == ".pyc" or "__pycache__" in path.parts:
                continue
            if path.name.startswith("."):
                continue
            files.append(path)
    return files


def check_privacy(paths: list[Path]) -> None:
    problems: list[str] = []
    for path in paths:
        rel = path.relative_to(ROOT)
        parts = set(rel.parts)
        if parts & FORBIDDEN_PARTS:
            problems.append(f"forbidden path part: {rel}")
        if rel.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden media or dataset suffix: {rel}")
        if rel.name in FORBIDDEN_NAMES:
            problems.append(f"personal file: {rel}")
    if problems:
        raise SystemExit("Release privacy check failed:\n" + "\n".join(problems))


def main() -> int:
    release_version = version()
    root_name = f"tractus-manus-{release_version}"
    DIST.mkdir(exist_ok=True)
    output = DIST / f"tractus-manus-{release_version}-windows.zip"
    manifest_path = DIST / f"tractus-manus-{release_version}-manifest.txt"
    checksum_path = DIST / f"tractus-manus-{release_version}-windows.zip.sha256"

    paths = collect()
    check_privacy(paths)
    config_example = (ROOT / "config.example.json").read_text(encoding="utf-8")
    if '"enabled": true' in config_example.split('"classifier":', 1)[-1].split("}", 1)[0]:
        raise SystemExit("config.example.json must keep the classifier disabled")

    entries: list[tuple[str, bytes]] = []
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        entries.append((f"{root_name}/{rel}", path.read_bytes()))
    entries.append((f"{root_name}/config.json", config_example.encode("utf-8")))
    entries.sort(key=lambda item: item[0])

    fixed_time = (2026, 10, 26, 0, 0, 0)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for arcname, data in entries:
            info = zipfile.ZipInfo(arcname, date_time=fixed_time)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o644 & 0xFFFF) << 16
            archive.writestr(info, data)

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    checksum_path.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    manifest_path.write_text(
        "\n".join(name for name, _ in entries) + "\n", encoding="utf-8"
    )

    print(f"Built {output}")
    print(f"Files {len(entries)}")
    print(f"Bytes {output.stat().st_size}")
    print(f"SHA256 {digest}")
    print(f"Manifest {manifest_path.name}")
    print(f"Checksum {checksum_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
