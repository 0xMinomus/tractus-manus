"""Test perekam data: format JSONL dan overlay preview."""

from __future__ import annotations

import json

import numpy as np

from handtrack.collector import draw_preview, write_sample
from tests.helpers import make_hand


def test_write_sample_roundtrip(tmp_path) -> None:
    path = tmp_path / "data.jsonl"
    hand = make_hand("point", label="Left")
    with path.open("w", encoding="utf-8") as stream:
        write_sample(stream, session="S1", label="neutral", hand=hand, now=1.25)

    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["session"] == "S1"
    assert record["label"] == "neutral"
    assert record["hand"] == "Left"
    assert record["t"] == 1.25
    landmarks = np.asarray(record["landmarks"])
    assert landmarks.shape == (21, 3)
    assert np.allclose(landmarks, hand.landmarks, atol=1e-6)


def test_draw_preview_runs() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    hand = make_hand("point")
    result = draw_preview(
        frame,
        [hand],
        session="S1",
        label="neutral",
        recording=True,
        counts={"neutral": 12},
        fps=30.0,
    )
    assert result.shape == frame.shape
    assert result.any()
