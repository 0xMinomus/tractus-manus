"""Konfigurasi aplikasi: dataclass bertipe + pemuatan JSON yang tervalidasi."""

from __future__ import annotations

import json
import logging
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

BACKENDS = {"dshow", "msmf", "any"}
HANDS = {"auto", "right", "left"}
MONITORS = {"primary", "virtual"}


class ConfigError(ValueError):
    """Nilai konfigurasi tidak masuk akal."""


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 60
    backend: str = "dshow"
    mirror: bool = True
    swap_handedness: bool = True


@dataclass(frozen=True)
class TrackingConfig:
    process_width: int = 800
    max_hands: int = 2
    min_detection_confidence: float = 0.6
    min_hand_presence_confidence: float = 0.55
    min_tracking_confidence: float = 0.55


@dataclass(frozen=True)
class SmoothingConfig:
    min_cutoff: float = 2.5
    beta: float = 7.0
    derivative_cutoff: float = 1.8


@dataclass(frozen=True)
class ActiveAreaConfig:
    x_min: float = 0.12
    x_max: float = 0.88
    y_min: float = 0.1
    y_max: float = 0.8


@dataclass(frozen=True)
class ControlConfig:
    primary_hand: str = "auto"
    target_monitor: str = "primary"
    click_anchor_ms: int = 60
    reacquire_ms: int = 150
    sensitivity: float = 1.2
    active_area: ActiveAreaConfig = field(default_factory=ActiveAreaConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)


@dataclass(frozen=True)
class ScrollConfig:
    enabled: bool = True
    enter_hold_ms: int = 120
    exit_hold_ms: int = 150
    step_px: int = 40
    smooth: float = 0.8
    invert: bool = False


@dataclass(frozen=True)
class PauseConfig:
    enabled: bool = True
    hold_ms: int = 1000


@dataclass(frozen=True)
class ClassifierConfig:
    enabled: bool = False
    model_path: str = "models/gesture_classifier.npz"
    confidence: float = 0.65
    smoothing: float = 0.6


@dataclass(frozen=True)
class GestureConfig:
    pinch_press: float = 0.36
    pinch_release: float = 0.52
    pinch_margin: float = 0.75
    pinch_stable_ms: int = 25
    click_cooldown_ms: int = 100
    left_hold_ms: int = 300
    middle_click_press: float = 0.18
    middle_click_release: float = 0.32
    left_click: bool = True
    right_click: bool = True
    middle_click: bool = True
    scroll: ScrollConfig = field(default_factory=ScrollConfig)
    pause: PauseConfig = field(default_factory=PauseConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)


@dataclass(frozen=True)
class UIConfig:
    show_preview: bool = True
    always_on_top: bool = True
    show_skeleton: bool = True
    debug: bool = False


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    ui: UIConfig = field(default_factory=UIConfig)


def _build(cls: type, data: Any, path: str, warnings: list[str]) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"'{path.rstrip('.')}' harus berupa objek JSON")
    allowed = {f.name for f in fields(cls)}
    for key in data:
        if key not in allowed:
            warnings.append(f"Opsi tidak dikenal: {path}{key}")

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        raw = data[f.name]
        if f.default_factory is not MISSING:  # type: ignore[misc]
            default = f.default_factory()
        elif f.default is not MISSING:
            default = f.default
        else:  # pragma: no cover - semua field punya default
            raise ConfigError(f"Field '{path}{f.name}' tidak memiliki nilai default")
        label = f"{path}{f.name}"

        if is_dataclass(default):
            kwargs[f.name] = _build(type(default), raw, f"{label}.", warnings)
        elif isinstance(default, bool):
            if not isinstance(raw, bool):
                raise ConfigError(f"'{label}' harus true/false")
            kwargs[f.name] = raw
        elif isinstance(default, int):
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise ConfigError(f"'{label}' harus bilangan bulat")
            kwargs[f.name] = raw
        elif isinstance(default, float):
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ConfigError(f"'{label}' harus angka")
            kwargs[f.name] = float(raw)
        elif isinstance(default, str):
            if not isinstance(raw, str):
                raise ConfigError(f"'{label}' harus teks")
            kwargs[f.name] = raw
        else:  # pragma: no cover - tipe konfigurasi selalu sederhana
            raise ConfigError(f"Tipe '{label}' tidak didukung")
    return cls(**kwargs)


def _validate(cfg: AppConfig) -> None:
    cam = cfg.camera
    if cam.index < 0:
        raise ConfigError("camera.index tidak boleh negatif")
    if not (320 <= cam.width <= 7680 and 240 <= cam.height <= 4320):
        raise ConfigError("camera.width/height di luar rentang wajar")
    if not (1 <= cam.fps <= 240):
        raise ConfigError("camera.fps harus 1..240")
    if cam.backend not in BACKENDS:
        raise ConfigError(f"camera.backend harus salah satu dari {sorted(BACKENDS)}")

    trk = cfg.tracking
    if trk.process_width != 0 and not (320 <= trk.process_width <= 1920):
        raise ConfigError("tracking.process_width harus 0 (native) atau 320..1920")
    if not (1 <= trk.max_hands <= 2):
        raise ConfigError("tracking.max_hands harus 1 atau 2")
    for name in (
        "min_detection_confidence",
        "min_hand_presence_confidence",
        "min_tracking_confidence",
    ):
        value = getattr(trk, name)
        if not (0.0 < value <= 1.0):
            raise ConfigError(f"tracking.{name} harus 0..1")

    ctl = cfg.control
    if ctl.primary_hand not in HANDS:
        raise ConfigError(f"control.primary_hand harus salah satu dari {sorted(HANDS)}")
    if ctl.target_monitor not in MONITORS:
        raise ConfigError(f"control.target_monitor harus salah satu dari {sorted(MONITORS)}")
    if not (0 <= ctl.click_anchor_ms <= 1000):
        raise ConfigError("control.click_anchor_ms harus 0..1000")
    if not (0 <= ctl.reacquire_ms <= 2000):
        raise ConfigError("control.reacquire_ms harus 0..2000")
    if not (0.5 <= ctl.sensitivity <= 3.0):
        raise ConfigError("control.sensitivity harus 0.5..3.0")

    area = ctl.active_area
    values = (area.x_min, area.x_max, area.y_min, area.y_max)
    if not all(0.0 <= v <= 1.0 for v in values):
        raise ConfigError("control.active_area harus di dalam 0..1")
    if area.x_max - area.x_min < 0.1 or area.y_max - area.y_min < 0.1:
        raise ConfigError("control.active_area terlalu sempit (minimal 10% layar)")
    if area.x_min >= area.x_max or area.y_min >= area.y_max:
        raise ConfigError("control.active_area: min harus lebih kecil dari max")

    sm = ctl.smoothing
    if not (0.05 <= sm.min_cutoff <= 20.0):
        raise ConfigError("control.smoothing.min_cutoff harus 0.05..20")
    if not (0.0 <= sm.beta <= 100.0):
        raise ConfigError("control.smoothing.beta harus 0..100")
    if not (0.05 <= sm.derivative_cutoff <= 20.0):
        raise ConfigError("control.smoothing.derivative_cutoff harus 0.05..20")

    g = cfg.gestures
    if not (0.02 <= g.pinch_press < g.pinch_release <= 1.5):
        raise ConfigError("gestures: butuh 0.02 <= pinch_press < pinch_release <= 1.5")
    if not (0.5 <= g.pinch_margin <= 1.0):
        raise ConfigError("gestures.pinch_margin harus 0.5..1.0")
    if not (0 <= g.pinch_stable_ms <= 200):
        raise ConfigError("gestures.pinch_stable_ms harus 0..200")
    if not (0 <= g.click_cooldown_ms <= 1000):
        raise ConfigError("gestures.click_cooldown_ms harus 0..1000")
    if not (0 <= g.left_hold_ms <= 2000):
        raise ConfigError("gestures.left_hold_ms harus 0..2000")
    if not (0.02 <= g.middle_click_press < g.middle_click_release <= 1.5):
        raise ConfigError(
            "gestures: butuh 0.02 <= middle_click_press < middle_click_release <= 1.5"
        )
    if not (g.left_click or g.right_click or g.middle_click):
        raise ConfigError("minimal satu tombol klik harus diaktifkan")

    s = g.scroll
    if not (0 <= s.enter_hold_ms <= 2000 and 0 <= s.exit_hold_ms <= 2000):
        raise ConfigError("gestures.scroll: enter/exit hold harus 0..2000 ms")
    if s.step_px < 5:
        raise ConfigError("gestures.scroll.step_px minimal 5")
    if not (0.05 <= s.smooth <= 1.0):
        raise ConfigError("gestures.scroll.smooth harus 0.05..1")

    if not (200 <= g.pause.hold_ms <= 5000):
        raise ConfigError("gestures.pause.hold_ms harus 200..5000")

    clf = g.classifier
    if not clf.model_path.strip():
        raise ConfigError("gestures.classifier.model_path tidak boleh kosong")
    if not (0.5 <= clf.confidence <= 1.0):
        raise ConfigError("gestures.classifier.confidence harus 0.5..1.0")
    if not (0.0 <= clf.smoothing <= 1.0):
        raise ConfigError("gestures.classifier.smoothing harus 0..1")


def load_config(path: Path | str) -> AppConfig:
    """Muat konfigurasi; field yang tidak ada memakai default."""
    path = Path(path)
    warnings: list[str] = []
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"{path}: JSON tidak valid (baris {exc.lineno}, kolom {exc.colno}): {exc.msg}"
            ) from exc
        except OSError as exc:
            raise ConfigError(f"Tidak bisa membaca {path}: {exc}") from exc
    else:
        raw = {}
    cfg = _build(AppConfig, raw, "", warnings)
    for message in warnings:
        log.warning(message)
    _validate(cfg)
    return cfg


def save_config(cfg: AppConfig, path: Path | str) -> None:
    path = Path(path)
    path.write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
