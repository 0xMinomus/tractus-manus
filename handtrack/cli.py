"""Entrypoint CLI Tractus Manus."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path

# Kurangi log internal TensorFlow Lite / MediaPipe sebelum import.
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from .app import HandTrackApp, RunOptions  # noqa: E402
from .config import AppConfig, ConfigError, load_config, save_config  # noqa: E402
from .winapi import enable_dpi_awareness  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.json"
VERSION = "1.0.0"

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tractus-manus",
        description="Kontrol mouse Windows memakai gestur tangan dari webcam.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"file konfigurasi JSON (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument("--camera", type=int, default=None, help="index kamera (0, 1, ...)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="jangan gerakkan mouse; cukup tampilkan aksi di log",
    )
    parser.add_argument(
        "--no-preview", action="store_true", help="jalankan tanpa jendela preview"
    )
    parser.add_argument("--debug", action="store_true", help="log debug + overlay debug")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        metavar="DETIK",
        help="berhenti otomatis setelah N detik (untuk pengujian)",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="tulis config default ke path --config jika belum ada, lalu keluar",
    )
    parser.add_argument("--version", action="version", version=f"Tractus Manus {VERSION}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path: Path = args.config
    if args.write_config:
        if config_path.exists():
            print(f"Config sudah ada: {config_path} (tidak ditimpa)")
        else:
            try:
                save_config(AppConfig(), config_path)
            except OSError as exc:
                log.error("Tidak bisa menulis config: %s", exc)
                return 1
            print(f"Config default ditulis ke {config_path}")
        return 0

    if args.duration is not None and args.duration <= 0:
        log.error("--duration harus lebih besar dari 0")
        return 2

    try:
        cfg = load_config(config_path)
        if args.camera is not None:
            if args.camera < 0:
                raise ConfigError("--camera tidak boleh negatif")
            cfg = replace(cfg, camera=replace(cfg.camera, index=args.camera))
        if not config_path.exists():
            log.warning("Config %s tidak ditemukan; memakai nilai default.", config_path)
        else:
            log.info("Config: %s", config_path)

        enable_dpi_awareness()
        options = RunOptions(
            dry_run=args.dry_run,
            duration=args.duration,
            preview=False if args.no_preview else None,
            debug=True if args.debug else None,
        )
        HandTrackApp(cfg, options).run()
        return 0
    except KeyboardInterrupt:
        print("\nDihentikan oleh pengguna. Semua tombol mouse dilepas.")
        return 0
    except ConfigError as exc:
        log.error("Konfigurasi tidak valid: %s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001 - pesan ramah ke pengguna
        if args.debug:
            log.exception("Terjadi kesalahan")
        else:
            log.error("%s", exc)
            log.error("Jalankan ulang dengan --debug untuk detail teknis.")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
