"""Test fitur invarian dan classifier NumPy (tanpa webcam)."""

from __future__ import annotations

import numpy as np
import pytest

from handtrack.classifier import (
    FEATURE_DIM,
    GestureClassifier,
    extract_features,
    extract_features_from_landmarks,
)
from tests.helpers import make_hand


def test_features_shape_and_pose_difference() -> None:
    point = extract_features(make_hand("point"))
    fist = extract_features(make_hand("fist", thumb=(480, 490)))
    assert point.shape == (FEATURE_DIM,)
    assert np.isfinite(point).all()
    assert not np.allclose(point, fist)


def test_features_mirror_left_right() -> None:
    right = make_hand("scroll", label="Right")
    mirrored = right.landmarks.copy()
    mirrored[:, 0] = 1.0 - mirrored[:, 0]
    left = extract_features_from_landmarks(mirrored, "Left")
    np.testing.assert_allclose(extract_features(right), left, atol=1e-6)


def test_features_rotation_scale_translation() -> None:
    base = make_hand("point", label="Right")
    expected = extract_features(base)

    wrist = base.landmarks[0, :2].copy()
    theta = np.deg2rad(37.0)
    rot = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
    rotated = base.landmarks.copy()
    rotated[:, :2] = (rotated[:, :2] - wrist) @ rot.T + wrist

    scaled = base.landmarks.copy()
    scaled[:, :2] = wrist + (scaled[:, :2] - wrist) * 1.6

    moved = base.landmarks.copy()
    moved[:, :2] += np.array((0.17, -0.11))

    for transformed in (rotated, scaled, moved):
        got = extract_features_from_landmarks(transformed, "Right")
        np.testing.assert_allclose(got, expected, atol=1e-5)


def _save_model(tmp_path, *, w1=None, classes=("neutral", "left")) -> GestureClassifier:
    n_out = len(classes)
    if w1 is None:
        w1 = np.zeros((FEATURE_DIM, 8), dtype=np.float32)
    path = tmp_path / "model.npz"
    np.savez(
        path,
        W1=w1,
        b1=np.zeros(8, dtype=np.float32),
        W2=np.zeros((8, 4), dtype=np.float32),
        b2=np.zeros(4, dtype=np.float32),
        W3=np.zeros((4, n_out), dtype=np.float32),
        b3=np.array([0.0, 10.0][:n_out] + [0.0] * max(0, n_out - 2), dtype=np.float32),
        mean=np.zeros(FEATURE_DIM, dtype=np.float32),
        std=np.ones(FEATURE_DIM, dtype=np.float32),
        classes=np.array(classes, dtype=str),
    )
    return GestureClassifier(path, smoothing=0.0)


def test_classifier_infers_and_resets(tmp_path) -> None:
    clf = _save_model(tmp_path)
    label, confidence = clf.predict(make_hand("point"))
    assert label == "left"
    assert confidence > 0.9
    clf.reset()
    assert clf._probs is None


def test_classifier_rejects_incompatible_shape(tmp_path) -> None:
    with pytest.raises(ValueError, match="tidak kompatibel"):
        _save_model(tmp_path, w1=np.zeros((10, 8), dtype=np.float32))
