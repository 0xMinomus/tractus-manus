"""Fitur tangan yang invarian + classifier NN NumPy untuk gestur halus.

Fitur (47 dimensi) memakai landmark yang sudah dikanonikalisasi:
- 20 titik (tanpa wrist) x/y, diskalakan dengan panjang telapak dan diputar
  supaya telapak menghadap ke atas -> 40 nilai
- 3 jarak pinch (telunjuk/tengah/manis ke ibu jari) -> 3 nilai
- 4 sudut kelengkungan jari (PIP) -> 4 nilai

Semua invarian terhadap posisi, skala, rotasi layar, dan tangan kiri/kanan.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol

import numpy as np

from .hands import (
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_PIP,
    PINKY_TIP,
    PROJECT_ROOT,
    RING_PIP,
    RING_TIP,
    THUMB_TIP,
    WRIST,
    Hand,
)

CLASSES: tuple[str, ...] = ("neutral", "left", "right", "middle", "fist", "scroll")
FEATURE_DIM = 47
DEFAULT_MODEL_PATH = "models/gesture_classifier.npz"


class Classifier(Protocol):
    def predict(self, hand: Hand) -> tuple[str, float]: ...

    def reset(self) -> None: ...


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Sudut di b (derajat), aman untuk vektor nol."""
    v1 = a - b
    v2 = c - b
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def canonical_points(landmarks: np.ndarray, label: str) -> np.ndarray:
    """(21,2) landmark ternormalisasi, wrist di origin, telapak ke atas, skala=1."""
    pts = np.asarray(landmarks, dtype=np.float64)[:, :2].copy()
    if label == "Left":
        pts[:, 0] = -pts[:, 0]
    pts -= pts[WRIST]
    palm = pts[MIDDLE_MCP]
    palm_len = float(np.linalg.norm(palm))
    if palm_len < 1e-9:
        return np.zeros_like(pts)
    # Putar supaya vektor wrist->middle MCP menunjuk ke (0,-1).
    theta = -math.pi / 2.0 - math.atan2(palm[1], palm[0])
    c, s = math.cos(theta), math.sin(theta)
    x, y = pts[:, 0].copy(), pts[:, 1].copy()
    pts[:, 0] = c * x - s * y
    pts[:, 1] = s * x + c * y
    return pts / palm_len


def extract_features_from_landmarks(landmarks: np.ndarray, label: str) -> np.ndarray:
    pts = canonical_points(landmarks, label)
    flat = pts[1:].reshape(-1)
    pinches = np.array(
        (
            np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP]),
            np.linalg.norm(pts[THUMB_TIP] - pts[MIDDLE_TIP]),
            np.linalg.norm(pts[THUMB_TIP] - pts[RING_TIP]),
        ),
        dtype=np.float64,
    )
    angles = np.array(
        (
            _angle(pts[5], pts[INDEX_PIP], pts[INDEX_TIP]),
            _angle(pts[9], pts[MIDDLE_PIP], pts[MIDDLE_TIP]),
            _angle(pts[13], pts[RING_PIP], pts[RING_TIP]),
            _angle(pts[17], pts[PINKY_PIP], pts[PINKY_TIP]),
        ),
        dtype=np.float64,
    )
    return np.concatenate((flat, pinches, angles)).astype(np.float32)


def extract_features(hand: Hand) -> np.ndarray:
    return extract_features_from_landmarks(hand.landmarks, hand.label)


def resolve_model_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


class GestureClassifier:
    """MLP 47 -> 128 -> 64 -> n, bobot dari file npz hasil trainer."""

    def __init__(self, model_path: str | Path, smoothing: float = 0.6) -> None:
        path = resolve_model_path(model_path)
        data = np.load(path, allow_pickle=False)
        try:
            self.w1 = data["W1"].astype(np.float32)
            self.b1 = data["b1"].astype(np.float32)
            self.w2 = data["W2"].astype(np.float32)
            self.b2 = data["b2"].astype(np.float32)
            self.w3 = data["W3"].astype(np.float32)
            self.b3 = data["b3"].astype(np.float32)
            self.mean = data["mean"].astype(np.float32)
            self.std = np.maximum(data["std"].astype(np.float32), 1e-6)
            self.classes = [str(c) for c in data["classes"].tolist()]
        finally:
            data.close()

        if self.w1.ndim != 2 or self.w1.shape[0] != FEATURE_DIM:
            raise ValueError(
                f"Model tidak kompatibel: W1 {self.w1.shape}, harus ({FEATURE_DIM}, hidden)."
            )
        if (
            self.w2.shape[0] != self.w1.shape[1]
            or self.w3.shape[0] != self.w2.shape[1]
            or self.b1.shape != (self.w1.shape[1],)
            or self.b2.shape != (self.w2.shape[1],)
            or self.b3.shape != (self.w3.shape[1],)
            or self.mean.shape != (FEATURE_DIM,)
            or self.std.shape != (FEATURE_DIM,)
            or len(self.classes) != self.w3.shape[1]
        ):
            raise ValueError("Model tidak kompatibel: dimensi layer/kelas tidak cocok.")
        self.smoothing = float(np.clip(smoothing, 0.0, 1.0))
        self._probs: np.ndarray | None = None

    def reset(self) -> None:
        self._probs = None

    def _infer(self, features: np.ndarray) -> np.ndarray:
        x = (np.asarray(features, dtype=np.float32) - self.mean) / self.std
        h1 = np.maximum(x @ self.w1 + self.b1, 0.0)
        h2 = np.maximum(h1 @ self.w2 + self.b2, 0.0)
        logits = h2 @ self.w3 + self.b3
        logits -= np.max(logits)
        e = np.exp(logits)
        return e / max(float(np.sum(e)), 1e-12)

    def predict(self, hand: Hand) -> tuple[str, float]:
        probs = self._infer(extract_features(hand))
        if self._probs is None or self.smoothing <= 0.0:
            self._probs = probs
        else:
            self._probs = self.smoothing * self._probs + (1.0 - self.smoothing) * probs
        idx = int(np.argmax(self._probs))
        return self.classes[idx], float(self._probs[idx])
