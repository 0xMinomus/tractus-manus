"""Perekam data latih gestur dari webcam.

Jalankan, pilih label dengan tombol 1-6, tahan pose, dan tekan SPACE untuk
mulai/berhenti merekam. Data JSONL dipakai train_gesture.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .camera import open_camera, resize_for_processing
from .classifier import CLASSES
from .config import AppConfig, ConfigError, load_config
from .hands import HAND_CONNECTIONS, PROJECT_ROOT, Hand, HandTracker

log = logging.getLogger(__name__)

DEFAULT_CONFIG = PROJECT_ROOT / "config.json"
DEFAULT_OUT = PROJECT_ROOT / "data" / "gestures.jsonl"
WINDOW = "HandTrack Collector"
LABEL_KEYS = {str(i + 1): label for i, label in enumerate(CLASSES)}
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (245, 245, 245)
MUTED = (170, 170, 170)
GREEN = (80, 220, 110)
RED = (70, 80, 235)
YELLOW = (70, 210, 235)


def draw_preview(
    frame: np.ndarray,
    hands: list[Hand],
    *,
    session: str,
    label: str,
    recording: bool,
    counts: Counter,
    fps: float,
) -> np.ndarray:
    h, w = frame.shape[:2]
    for hand in hands:
        points = [(int(x * w), int(y * h)) for x, y in hand.landmarks[:, :2]]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, points[a], points[b], (60, 140, 220), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, point, 4, WHITE, -1, cv2.LINE_AA)

    lines = [
        (f"SESI {session}", WHITE),
        (f"LABEL {label.upper()}  [1-6]", YELLOW),
        ("REC  space = stop" if recording else "PAUSE  space = rekam", RED if recording else MUTED),
        (
            f"{counts.get(label, 0)} frame label ini  |  "
            f"tangan={hands[0].label if len(hands) == 1 else len(hands)}  |  {fps:.0f} fps",
            MUTED,
        ),
    ]
    if len(hands) != 1:
        lines.append(("Tunjukkan SATU tangan saja", RED))
    else:
        lines.append((f"confidence {hands[0].score:.2f}", MUTED))

    y = 30
    for text, color in lines:
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.62, 1)
        cv2.rectangle(frame, (12, y - 22), (12 + tw + 16, y + 8), (25, 25, 25), -1)
        cv2.putText(frame, text, (20, y), FONT, 0.62, color, 1, cv2.LINE_AA)
        y += 32
    cv2.putText(
        frame,
        "1-6 label | space rekam | n sesi baru | q keluar",
        (20, h - 18),
        FONT,
        0.5,
        MUTED,
        1,
        cv2.LINE_AA,
    )
    return frame


def write_sample(stream, *, session: str, label: str, hand: Hand, now: float) -> None:
    record = {
        "session": session,
        "label": label,
        "hand": hand.label,
        "landmarks": np.round(hand.landmarks, 6).tolist(),
        "t": round(now, 3),
    }
    stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    stream.flush()


def run(args: argparse.Namespace) -> int:
    cfg: AppConfig = load_config(args.config)
    if args.camera is not None:
        if args.camera < 0:
            raise ConfigError("--camera tidak boleh negatif")
        cfg = replace(cfg, camera=replace(cfg.camera, index=args.camera))

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    session_no = 1
    session = f"S{datetime.now():%Y%m%d-%H%M%S}-{session_no}"
    label = CLASSES[0]
    recording = False
    counts: Counter[str] = Counter()
    fps = 0.0
    last_at = time.perf_counter()
    frames = 0

    cap = None
    tracker = None
    try:
        cap = open_camera(cfg.camera)
        tracker = HandTracker(
            cfg.tracking,
            mirror=cfg.camera.mirror,
            swap_handedness=cfg.camera.swap_handedness,
        )
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        print()
        print("=" * 62)
        print("  Perekam data gestur HandTrack")
        print("=" * 62)
        print(f"  Output : {out_path}")
        print(f"  Sesi   : {session}")
        print("  1-6    : pilih label (merekam otomatis berhenti)")
        print("  space  : mulai / berhenti merekam")
        print("  n      : sesi baru (ulangi semua label untuk sesi ini)")
        print("  q/ESC  : keluar")
        print()
        print("  Rekam tiap label 20-40 detik, tangan bervariasi jarak/posisi.")
        print("  Lakukan minimal 2 sesi, masing-masing berisi semua label.")
        print("=" * 62)

        with out_path.open("a", encoding="utf-8") as stream:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    log.error("Frame kamera tidak terbaca; keluar.")
                    return 1
                frame = resize_for_processing(frame, cfg.tracking.process_width)
                display, hands = tracker.process(frame)

                now = time.perf_counter()
                frames += 1
                if now - last_at >= 0.5:
                    fps = frames / (now - last_at)
                    frames = 0
                    last_at = now

                if recording and len(hands) == 1 and hands[0].score >= 0.6:
                    write_sample(
                        stream,
                        session=session,
                        label=label,
                        hand=hands[0],
                        now=now,
                    )
                    counts[label] += 1

                draw_preview(
                    display,
                    hands,
                    session=session,
                    label=label,
                    recording=recording,
                    counts=counts,
                    fps=fps,
                )
                cv2.imshow(WINDOW, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if ord("1") <= key <= ord("6"):
                    new_label = LABEL_KEYS[chr(key)]
                    if new_label != label:
                        label = new_label
                        recording = False
                        log.info("Label: %s (rekaman dijeda)", label)
                elif key == ord(" "):
                    recording = not recording
                    log.info("Rekam %s", "mulai" if recording else "berhenti")
                elif key in (ord("n"), ord("N")):
                    session_no += 1
                    session = f"S{datetime.now():%Y%m%d-%H%M%S}-{session_no}"
                    recording = False
                    counts = Counter()
                    log.info("Sesi baru: %s", session)
        return 0
    finally:
        if cap is not None:
            cap.release()
        if tracker is not None:
            tracker.close()
        cv2.destroyAllWindows()
        print(f"\nData tersimpan di {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="collect_hand_data",
        description="Rekam data landmark tangan untuk melatih classifier gestur.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"file JSONL (default: {DEFAULT_OUT})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\nPerekaman dibatalkan.")
        return 0
    except (ConfigError, RuntimeError) as exc:
        log.error("%s", exc)
        return 2 if isinstance(exc, ConfigError) else 1
    except Exception as exc:  # noqa: BLE001 - pesan ramah ke pengguna
        log.exception("Gagal merekam data: %s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
