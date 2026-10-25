"""Test loader dataset, trainer, dan classifier hasil training."""

from __future__ import annotations

import json

import numpy as np
import pytest

from handtrack.classifier import CLASSES, FEATURE_DIM, GestureClassifier
from handtrack.training import Dataset, load_dataset, train_model
from tests.helpers import make_hand


def test_load_dataset(tmp_path) -> None:
    path = tmp_path / "data.jsonl"
    hand = make_hand("point")
    records = [
        {
            "session": "S1",
            "label": "neutral",
            "hand": hand.label,
            "landmarks": hand.landmarks.tolist(),
        },
        {
            "session": "S2",
            "label": "left",
            "hand": hand.label,
            "landmarks": hand.landmarks.tolist(),
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    dataset = load_dataset(path)
    assert dataset.features.shape == (2, FEATURE_DIM)
    assert dataset.labels.tolist() == [0, 1]
    assert dataset.classes == ("neutral", "left")
    assert dataset.counts == {"neutral": 1, "left": 1}
    assert set(dataset.sessions) == {"S1", "S2"}


def test_load_dataset_rejects_bad_landmarks(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(
            {
                "session": "S1",
                "label": "left",
                "hand": "Right",
                "landmarks": [[0.0, 0.0, 0.0]],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="21,3"):
        load_dataset(path)


def test_train_model_and_load_npz(tmp_path) -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(7)
    sessions = ("S1", "S2")
    n_per = 30
    features = []
    labels = []
    session_labels = []
    for session in sessions:
        for index in range(len(CLASSES)):
            center = np.zeros(FEATURE_DIM, dtype=np.float32)
            center[index] = 3.0
            center[index + 7] = 1.5
            block = rng.normal(0.0, 0.5, size=(n_per, FEATURE_DIM)).astype(np.float32)
            features.append(block + center)
            labels.extend([index] * n_per)
            session_labels.extend([session] * n_per)
    dataset = Dataset(
        features=np.vstack(features),
        labels=np.array(labels, dtype=np.int64),
        sessions=np.array(session_labels),
        classes=CLASSES,
        counts={name: 2 * n_per for name in CLASSES},
    )

    report = train_model(
        dataset,
        tmp_path / "gesture_classifier.npz",
        epochs=20,
        batch_size=64,
        lr=0.01,
        seed=1,
        val_fraction=0.5,
        patience=5,
    )
    assert report.out_path.exists()
    assert report.val_accuracy > 0.8
    assert int(report.confusion.sum()) == 6 * n_per

    clf = GestureClassifier(report.out_path, smoothing=0.0)
    probs = clf._infer(np.asarray(dataset.features[0], dtype=np.float32))
    assert probs.shape == (len(CLASSES),)
    assert np.isfinite(probs).all()
    assert abs(float(probs.sum()) - 1.0) < 1e-5
    assert clf.classes == list(CLASSES)
