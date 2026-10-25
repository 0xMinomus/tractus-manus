"""Model tangan, fitur gestur, pemilihan tangan aktif, dan pembungkus MediaPipe."""

from __future__ import annotations

import hashlib
import logging
import math
import time
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from .config import TrackingConfig

log = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20
CURSOR_LANDMARK = INDEX_PIP

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

EXTENDED_ANGLE = 125.0


class Pinch(Enum):
    INDEX = "telunjuk"
    MIDDLE = "tengah"
    RING = "manis"


@dataclass(frozen=True)
class Hand:
    label: str  # "Left" / "Right" (tangan pengguna, bukan cermin)
    score: float
    landmarks: np.ndarray  # (21, 3), x/y ternormalisasi
    width: int
    height: int

    def point(self, index: int) -> np.ndarray:
        """Landmark dalam piksel frame."""
        lm = self.landmarks[index]
        return np.array((lm[0] * self.width, lm[1] * self.height), dtype=np.float64)

    def norm(self, index: int) -> tuple[float, float]:
        lm = self.landmarks[index]
        return float(lm[0]), float(lm[1])

    @property
    def palm_size(self) -> float:
        return float(np.linalg.norm(self.point(WRIST) - self.point(MIDDLE_MCP)))

    def dist(self, a: int, b: int) -> float:
        return float(np.linalg.norm(self.point(a) - self.point(b)))


def finger_angle(hand: Hand, mcp: int, pip: int, tip: int) -> float:
    a = hand.point(mcp) - hand.point(pip)
    b = hand.point(tip) - hand.point(pip)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return 0.0
    cos = float(np.dot(a, b) / denom)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


@dataclass(frozen=True)
class HandFeatures:
    pinch: dict[Pinch, float]
    angles: dict[str, float]
    scroll: bool
    fist: bool
    closest_pinch: Pinch | None
    closest_distance: float
    extended: dict[str, bool]
    classifier_pinch: Pinch | None = None


def analyze_hand(hand: Hand) -> HandFeatures:
    """Hitung jarak pinch ternormalisasi dan pose yang dipakai state machine."""
    palm = hand.palm_size
    if palm < 5.0:
        return HandFeatures({}, {}, False, False, None, float("inf"), {})

    pinch = {
        Pinch.INDEX: hand.dist(THUMB_TIP, INDEX_TIP) / palm,
        Pinch.MIDDLE: hand.dist(THUMB_TIP, MIDDLE_TIP) / palm,
        Pinch.RING: hand.dist(THUMB_TIP, RING_TIP) / palm,
    }
    closest = min(pinch, key=pinch.__getitem__)
    closest_d = pinch[closest]

    joints = {
        "index": (INDEX_MCP, INDEX_PIP, INDEX_TIP),
        "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
        "ring": (RING_MCP, RING_PIP, RING_TIP),
        "pinky": (PINKY_MCP, PINKY_PIP, PINKY_TIP),
    }
    angles: dict[str, float] = {}
    extended: dict[str, bool] = {}
    for name, (mcp, pip, tip) in joints.items():
        angle = finger_angle(hand, mcp, pip, tip)
        angles[name] = angle
        # Ujung harus lebih jauh dari pergelangan daripada sendi tengah.
        tip_far = hand.dist(tip, WRIST) > hand.dist(pip, WRIST) * 1.02
        extended[name] = angle >= EXTENDED_ANGLE and tip_far

    fist = all(angle < 100.0 for angle in angles.values())
    # Scroll = dua jari (telunjuk+tengah) terentang dan minimal satu dari
    # jari manis/kelingking terlipat. Ini menerima pose V dengan kelingking
    # sedikit keluar, tetapi tidak menganggap telapak terbuka sebagai scroll.
    folded_one = angles["ring"] < 110.0 or angles["pinky"] < 110.0
    scroll_pose = (
        extended["index"]
        and extended["middle"]
        and folded_one
        and not fist
    )
    return HandFeatures(
        pinch=pinch,
        angles=angles,
        scroll=scroll_pose,
        fist=fist,
        closest_pinch=closest,
        closest_distance=closest_d,
        extended=extended,
    )


def select_active_hand(hands: list[Hand], preference: str) -> Hand | None:
    """Pilih satu tangan pengendali; kanan diutamakan pada mode auto."""
    if not hands:
        return None
    wanted = "Left" if preference == "left" else "Right"
    return min(hands, key=lambda h: (0 if h.label == wanted else 1, -h.score))


def swap_label(label: str) -> str:
    return "Left" if label == "Right" else "Right"


def ensure_model(path: Path = DEFAULT_MODEL_PATH, timeout: float = 30.0) -> Path:
    """Pastikan model MediaPipe tersedia; unduh + verifikasi sekali jika tidak ada."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".download")
    log.info("Model belum ada, mengunduh %s ...", MODEL_URL)
    try:
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "handtrack/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if digest != MODEL_SHA256:
            raise OSError(f"hash model tidak cocok ({digest})")
        tmp.replace(path)
    except Exception as exc:  # noqa: BLE001 - pesan ke pengguna
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Gagal mengunduh model MediaPipe ke {path}. "
            "Periksa koneksi internet, atau unduh manual dari:\n"
            f"  {MODEL_URL}\n"
            f"lalu simpan sebagai {path}.\nDetail: {exc}"
        ) from exc
    log.info("Model tersimpan di %s", path)
    return path


class HandTracker:
    """Deteksi 1-2 tangan dengan MediaPipe HandLandmarker (mode VIDEO)."""

    def __init__(
        self, cfg: TrackingConfig, mirror: bool = True, swap_handedness: bool = False
    ) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model = ensure_model()
        self._mirror = mirror
        self.swap_handedness = swap_handedness
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=cfg.max_hands,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_hand_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def process(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, list[Hand]]:
        """Kembalikan frame tampilan (sudah dicerminkan) + tangan terdeteksi."""
        import mediapipe as mp

        display = cv2.flip(frame_bgr, 1) if self._mirror else frame_bgr
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts_ms = int(time.perf_counter() * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms
        result = self._landmarker.detect_for_video(image, ts_ms)

        height, width = display.shape[:2]
        hands: list[Hand] = []
        for index, landmarks in enumerate(result.hand_landmarks):
            handedness = result.handedness[index] if index < len(result.handedness) else []
            if handedness:
                category = handedness[0]
                label = category.category_name or "Right"
                score = float(category.score)
            else:
                label, score = "Right", 0.0
            if not self._mirror:
                label = swap_label(label)
            if self.swap_handedness:
                label = swap_label(label)
            points = np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float64)
            hands.append(Hand(label, score, points, width, height))
        return display, hands

    def close(self) -> None:
        self._landmarker.close()
