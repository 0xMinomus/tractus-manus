"""Unit test konfigurasi JSON."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import pytest

from handtrack.config import AppConfig, ConfigError, load_config, save_config

ROOT = Path(__file__).resolve().parents[1]


def test_defaults_when_missing(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "missing.json")
    assert cfg.camera.index == 0
    assert cfg.gestures.scroll.enabled
    assert cfg.control.primary_hand == "auto"


def test_repo_config_is_valid() -> None:
    cfg = load_config(ROOT / "config.json")
    assert cfg == AppConfig()
    assert cfg.camera.width == 1280
    assert cfg.gestures.middle_click_press < cfg.gestures.middle_click_release


def test_partial_merge(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "camera": {"index": 2},
                "gestures": {
                    "pinch_press": 0.22,
                    "pinch_release": 0.4,
                    "middle_click_press": 0.1,
                    "middle_click_release": 0.2,
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.camera.index == 2
    assert cfg.camera.width == 1280  # default
    assert cfg.gestures.pinch_press == pytest.approx(0.22)
    assert cfg.gestures.scroll.step_px == 40  # default nested


def test_unknown_key_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"camera": {"indx": 1}}), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        cfg = load_config(path)
    assert cfg.camera.index == 0
    assert "indx" in caplog.text


def test_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{ oops", encoding="utf-8")
    with pytest.raises(ConfigError, match="JSON"):
        load_config(path)


def test_wrong_type(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"camera": {"index": "nol"}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="bilangan bulat"):
        load_config(path)


def test_invalid_active_area(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"control": {"active_area": {"x_min": 0.9, "x_max": 0.1}}}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="active_area"):
        load_config(path)


def test_invalid_pinch_order(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"gestures": {"pinch_press": 0.8, "pinch_release": 0.2}}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="pinch_press"):
        load_config(path)


def test_invalid_middle_order(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"gestures": {"middle_click_press": 0.4, "middle_click_release": 0.2}}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="middle_click_press"):
        load_config(path)


def test_invalid_pinch_margin(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"gestures": {"pinch_margin": 0.2}}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="pinch_margin"):
        load_config(path)


def test_invalid_sensitivity(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"control": {"sensitivity": 9.0}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="sensitivity"):
        load_config(path)


def test_save_and_reload(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    save_config(AppConfig(), path)
    loaded = load_config(path)
    assert asdict(loaded) == asdict(AppConfig())
