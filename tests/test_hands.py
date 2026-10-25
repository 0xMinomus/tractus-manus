"""Unit test fitur tangan dan pemilihan tangan aktif."""

from __future__ import annotations

import pytest

from handtrack.hands import Pinch, analyze_hand, select_active_hand, swap_label
from tests.helpers import make_hand


def test_swap_label() -> None:
    assert swap_label("Right") == "Left"
    assert swap_label("Left") == "Right"


def test_open_hand_is_neither_scroll_nor_fist() -> None:
    hand = make_hand("open")
    assert hand.palm_size == pytest.approx(135.0)
    features = analyze_hand(hand)
    assert not features.scroll
    assert not features.fist
    assert features.angles["index"] > 170
    assert features.angles["pinky"] > 170
    assert features.closest_pinch is not None
    assert features.closest_distance > 1.0


def test_folded_finger_not_extended() -> None:
    hand = make_hand("point")
    features = analyze_hand(hand)
    assert features.extended["index"]
    assert not features.extended["middle"]
    assert not features.extended["ring"]
    assert not features.extended["pinky"]
    assert not features.scroll


def test_scroll_pose_ignores_thumb_ring_contact() -> None:
    hand = make_hand("scroll", thumb=(515, 470))
    features = analyze_hand(hand)
    assert features.scroll
    assert features.pinch[Pinch.RING] < 0.36
    assert features.pinch[Pinch.RING] > 0.18


def test_scroll_pose_allows_extended_pinky() -> None:
    hand = make_hand("scroll_pinky")
    features = analyze_hand(hand)
    assert features.scroll


def test_open_hand_does_not_scroll() -> None:
    features = analyze_hand(make_hand("open"))
    assert not features.scroll


def test_pinch_distances() -> None:
    hand = make_hand("open", pinch="index")
    features = analyze_hand(hand)
    assert features.pinch[Pinch.INDEX] == pytest.approx(0.0, abs=1e-9)
    assert features.closest_pinch is Pinch.INDEX


def test_select_active_hand() -> None:
    right = make_hand("open", label="Right")
    left = make_hand("open", label="Left")
    assert select_active_hand([], "auto") is None
    assert select_active_hand([left, right], "auto") is right
    assert select_active_hand([left, right], "right") is right
    assert select_active_hand([left, right], "left") is left
    assert select_active_hand([left], "right") is left
