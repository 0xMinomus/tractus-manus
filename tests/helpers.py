"""Helper landmark sintetis untuk unit test gestur."""

from __future__ import annotations

import numpy as np

from handtrack.hands import Hand

# Tangan menghadap kamera, jari ke atas. Satuan piksel; palm size 135 px.
_OPEN = {
    0: (500, 600),  # wrist
    1: (470, 560),
    2: (445, 530),
    3: (430, 495),
    4: (430, 460),  # thumb tip
    5: (455, 470), 6: (455, 400), 7: (455, 350), 8: (455, 260),
    9: (500, 465), 10: (500, 395), 11: (500, 345), 12: (500, 295),
    13: (545, 470), 14: (545, 405), 15: (545, 355), 16: (545, 305),
    17: (585, 485), 18: (585, 430), 19: (585, 390), 20: (585, 350),
}

# (mcp, pip, dip, tip)
_FINGERS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


def make_hand(
    pose: str = "point",
    pinch: str | None = None,
    label: str = "Right",
    thumb: tuple[float, float] | None = None,
    dx: float = 0.0,
    dy: float = 0.0,
) -> Hand:
    points = {index: (float(x), float(y)) for index, (x, y) in _OPEN.items()}

    folds = {
        "open": (),
        "point": ("middle", "ring", "pinky"),
        "scroll": ("ring", "pinky"),
        "scroll_pinky": ("ring",),
        "fist": ("index", "middle", "ring", "pinky"),
    }[pose]
    for name in folds:
        _mcp, pip, dip, tip = _FINGERS[name]
        points[dip] = (points[pip][0], points[pip][1] + 55.0)
        points[tip] = (points[pip][0], points[pip][1] + 90.0)

    if pinch is not None:
        tip_index = {"index": 8, "middle": 12, "ring": 16}[pinch]
        points[4] = points[tip_index]
    if thumb is not None:
        points[4] = (float(thumb[0]), float(thumb[1]))

    array = np.array([points[i] for i in range(21)], dtype=np.float64)
    array[:, 0] += dx
    array[:, 1] += dy
    landmarks = np.zeros((21, 3), dtype=np.float64)
    landmarks[:, :2] = array / 1000.0
    return Hand(label=label, score=0.99, landmarks=landmarks, width=1000, height=1000)


def make_near_hand(
    pose: str = "point",
    near_distance: float = 0.30,
    label: str = "Right",
    **kwargs,
) -> Hand:
    """Tangan dengan jempol dekat (tetapi tidak menempel) ujung telunjuk.

    Dipakai untuk gestur hold: jarak dinormalisasi panjang telapak.
    """
    base = make_hand(
        pose, label=label, **{k: v for k, v in kwargs.items() if k not in ("dx", "dy")}
    )
    tip = base.point(8)
    palm = base.palm_size
    thumb = (tip[0] + near_distance * palm, tip[1])
    return make_hand(pose, thumb=thumb, label=label, **kwargs)
