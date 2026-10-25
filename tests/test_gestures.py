"""Unit test state machine gestur (tanpa kamera)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from handtrack.config import GestureConfig, PauseConfig, ScrollConfig
from handtrack.gestures import Action, Freeze, GestureEngine, Mode
from handtrack.hands import Pinch, analyze_hand
from tests.helpers import make_hand


@pytest.fixture
def cfg() -> GestureConfig:
    return GestureConfig(
        scroll=ScrollConfig(enter_hold_ms=100, exit_hold_ms=100, step_px=50, smooth=1.0),
        pause=PauseConfig(hold_ms=500),
        pinch_stable_ms=0,
    )


def unlock(engine: GestureEngine, t: float = 0.2) -> None:
    """Tangan pertama mengunci gestur; pose netral membukanya."""
    engine.update([make_hand("point")], t - 0.1)
    state = engine.update([make_hand("point")], t)
    assert not state.locked


class FakeClassifier:
    def __init__(self, label: str = "neutral", confidence: float = 0.95) -> None:
        self.label = label
        self.confidence = confidence
        self.resets = 0

    def predict(self, hand) -> tuple[str, float]:
        return self.label, self.confidence

    def reset(self) -> None:
        self.resets += 1


def test_classifier_pinch_releases_when_label_changes(cfg: GestureConfig) -> None:
    fake = FakeClassifier("neutral")
    engine = GestureEngine(cfg, classifier=fake, classifier_confidence=0.6)
    unlock(engine)

    fake.label, fake.confidence = "left", 0.9
    pinched = make_hand("point", pinch="index")
    pressed = engine.update([pinched], 0.3)
    assert pressed.actions == [Action.LEFT_DOWN]
    assert pressed.ai_label == "left"
    assert pressed.ai_confidence == 0.9

    # Geometri masih jepit, tetapi AI sudah netral: tombol wajib dilepas.
    fake.label = "neutral"
    released = engine.update([pinched], 0.5)
    assert released.actions == [Action.LEFT_UP]
    assert released.ai_label == "neutral"


def test_classifier_reset_when_hand_lost(cfg: GestureConfig) -> None:
    fake = FakeClassifier("left")
    engine = GestureEngine(cfg, classifier=fake, classifier_confidence=0.6)
    engine.update([make_hand("point")], 0.1)
    engine.update([make_hand("point")], 0.2)
    state = engine.update([], 0.3)
    assert fake.resets == 1
    assert state.ai_label is None
    assert state.has_hand is False


def test_classifier_scroll_overrides_geometry(cfg: GestureConfig) -> None:
    fake = FakeClassifier("neutral")
    engine = GestureEngine(cfg, classifier=fake, classifier_confidence=0.6)
    unlock(engine)

    fake.label, fake.confidence = "scroll", 0.9
    hand = make_hand("point")  # geometri bukan pose scroll
    engine.update([hand], 0.3)
    state = engine.update([hand], 0.45)
    assert Action.SCROLL_START in state.actions
    assert state.mode is Mode.SCROLL


def test_classifier_fist_overrides_geometry(cfg: GestureConfig) -> None:
    fake = FakeClassifier("neutral")
    engine = GestureEngine(cfg, classifier=fake, classifier_confidence=0.6)
    unlock(engine)

    fake.label, fake.confidence = "fist", 0.9
    hand = make_hand("open")
    engine.update([hand], 0.3)
    state = engine.update([hand], 0.9)
    assert Action.PAUSE in state.actions
    assert state.mode is Mode.PAUSED


def test_feature_poses() -> None:
    open_hand = analyze_hand(make_hand("open"))
    assert not open_hand.scroll and not open_hand.fist

    scroll = analyze_hand(make_hand("scroll"))
    assert scroll.scroll and not scroll.fist

    fist = analyze_hand(make_hand("fist", thumb=(480, 490)))
    assert fist.fist and not fist.scroll

    # Pose V nyata: jempol menyentuh jari manis yang terlipat (0.27 palm).
    vsign = analyze_hand(make_hand("scroll", thumb=(515, 470)))
    assert vsign.scroll
    assert 0.18 < vsign.pinch[Pinch.RING] < 0.36


def test_left_click_hold_and_release(cfg: GestureConfig) -> None:
    engine = GestureEngine(cfg)
    unlock(engine)
    pinched = make_hand("point", pinch="index")

    state = engine.update([pinched], 0.3)
    assert state.actions == [Action.LEFT_DOWN]
    assert state.freeze is Freeze.ANCHOR
    assert state.pinch is not None and state.pinch.active

    # Tahan lebih dari 1 detik: tombol kiri tetap turun (hold/drag).
    for t in (0.5, 0.8, 1.2):
        held = engine.update([pinched], t)
        assert held.actions == []
        assert held.pinch is not None and held.pinch.active

    released = engine.update([make_hand("point")], 1.3)
    assert released.actions == [Action.LEFT_UP]
    assert released.freeze is Freeze.RELEASE
    assert released.pinch is not None and not released.pinch.active


def test_right_click_on_release(cfg: GestureConfig) -> None:
    engine = GestureEngine(cfg)
    unlock(engine)
    pinched = make_hand("point", pinch="middle")

    state = engine.update([pinched], 0.3)
    assert state.actions == []
    assert state.freeze is Freeze.HOLD

    released = engine.update([make_hand("open")], 0.5)
    assert released.actions == [Action.RIGHT_CLICK]
    assert released.freeze is Freeze.RELEASE


def test_middle_click_uses_tight_ring_threshold(cfg: GestureConfig) -> None:
    engine = GestureEngine(cfg)
    unlock(engine)

    # V-sign dengan jempol menempel jari manis TIDAK boleh jadi klik tengah.
    vsign = make_hand("scroll", thumb=(515, 470))
    first = engine.update([vsign], 0.3)
    assert first.actions == []
    arming = engine.update([vsign], 0.45)
    assert arming.actions == [Action.SCROLL_START]
    assert arming.mode is Mode.SCROLL

    engine2 = GestureEngine(cfg)
    unlock(engine2)
    ring_pinch = make_hand("open", pinch="ring")
    state = engine2.update([ring_pinch], 0.3)
    assert state.actions == []
    assert state.freeze is Freeze.HOLD
    released = engine2.update([make_hand("open")], 0.5)
    assert released.actions == [Action.MIDDLE_CLICK]


def test_fist_never_clicks(cfg: GestureConfig) -> None:
    engine = GestureEngine(replace(cfg, pause=PauseConfig(enabled=False)))
    unlock(engine)
    fist = make_hand("fist", thumb=(480, 490))
    for t in (0.3, 0.6, 1.2):
        state = engine.update([fist], t)
        assert state.actions == []
        assert state.pinch is None or not state.pinch.active


def test_hand_loss_releases_left_button(cfg: GestureConfig) -> None:
    engine = GestureEngine(cfg)
    unlock(engine)
    engine.update([make_hand("point", pinch="index")], 0.3)
    lost = engine.update([], 0.4)
    assert lost.actions == [Action.LEFT_UP]
    assert lost.freeze is Freeze.HOLD
    assert not lost.has_hand
    assert lost.mode is Mode.POINT


def test_right_pinch_loss_does_not_click(cfg: GestureConfig) -> None:
    engine = GestureEngine(cfg)
    unlock(engine)
    engine.update([make_hand("point", pinch="middle")], 0.3)
    lost = engine.update([], 0.4)
    assert lost.actions == []
    assert lost.freeze is Freeze.HOLD


def test_scroll_wheel_and_exit(cfg: GestureConfig) -> None:
    engine = GestureEngine(cfg)
    unlock(engine)

    state = engine.update([make_hand("scroll")], 0.3)
    assert state.actions == []
    state = engine.update([make_hand("scroll")], 0.45)
    assert state.actions == [Action.SCROLL_START]
    assert state.mode is Mode.SCROLL
    assert state.cursor is None
    assert state.freeze is Freeze.HOLD

    up = engine.update([make_hand("scroll", dy=-60)], 0.5)
    assert up.actions == [Action.WHEEL_UP]
    back = engine.update([make_hand("scroll")], 0.55)
    assert back.actions == [Action.WHEEL_DOWN]

    still = engine.update([make_hand("point")], 0.6)
    assert still.actions == []
    assert still.mode is Mode.SCROLL
    ended = engine.update([make_hand("point")], 0.75)
    assert ended.actions == [Action.SCROLL_END]
    assert ended.mode is Mode.POINT
    assert ended.freeze is Freeze.RELEASE

    resumed = engine.update([make_hand("point")], 0.9)
    assert resumed.cursor is not None
    assert resumed.mode is Mode.POINT


def test_scroll_exits_on_pinch_without_click(cfg: GestureConfig) -> None:
    engine = GestureEngine(cfg)
    unlock(engine)
    engine.update([make_hand("scroll")], 0.3)
    engine.update([make_hand("scroll")], 0.45)

    pinched = engine.update([make_hand("scroll", pinch="index")], 0.55)
    assert Action.SCROLL_END in pinched.actions
    assert Action.LEFT_DOWN not in pinched.actions
    assert pinched.mode is Mode.POINT
    assert pinched.locked

    # Pinch masih ditahan: harus tetap terkunci (tidak ada klik tak sengaja).
    still = engine.update([make_hand("scroll", pinch="index")], 0.65)
    assert still.actions == []
    assert still.locked

    # Buka ke pose netral, lalu pinch baru baru boleh klik.
    engine.update([make_hand("point")], 0.75)
    click = engine.update([make_hand("point", pinch="index")], 0.85)
    assert click.actions == [Action.LEFT_DOWN]


def test_open_palm_does_not_pause(cfg: GestureConfig) -> None:
    engine = GestureEngine(cfg)
    unlock(engine, 0.2)
    for t in (0.3, 0.8, 1.3, 2.0):
        state = engine.update([make_hand("open")], t)
        assert state.actions == []
        assert state.mode is Mode.POINT
        assert state.pause_progress == 0.0


def test_pause_cycle(cfg: GestureConfig) -> None:
    engine = GestureEngine(cfg)
    unlock(engine, 0.2)
    fist = make_hand("fist", thumb=(480, 490))

    state = engine.update([fist], 0.3)
    assert state.pose == "fist"
    state = engine.update([fist], 0.5)
    assert 0 < state.pause_progress < 1
    state = engine.update([fist], 0.85)
    assert state.actions == [Action.PAUSE]
    assert state.mode is Mode.PAUSED
    assert state.freeze is Freeze.HOLD

    # Kepalan yang sama terus ditahan: tidak boleh langsung lanjut.
    still = engine.update([fist], 1.0)
    assert still.actions == []
    assert still.mode is Mode.PAUSED
    still = engine.update([fist], 1.5)
    assert still.actions == []
    assert still.mode is Mode.PAUSED

    # Pinch saat jeda juga tidak menghasilkan aksi.
    paused_pinch = engine.update([make_hand("point", pinch="index")], 1.6)
    assert paused_pinch.actions == []
    assert paused_pinch.mode is Mode.PAUSED

    # Lepas kepalan, lalu tahan lagi untuk lanjut.
    engine.update([make_hand("point")], 1.7)
    engine.update([fist], 1.8)
    state = engine.update([fist], 2.4)
    assert state.actions == [Action.RESUME]
    assert state.mode is Mode.POINT
    assert state.freeze is Freeze.RELEASE

    # Kepalan masih ditahan setelah lanjut: jangan langsung jeda lagi.
    engine.update([fist], 2.6)
    engine.update([fist], 3.3)
    assert engine.mode is Mode.POINT

    engine.update([make_hand("point")], 3.4)
    engine.update([fist], 3.5)
    state = engine.update([fist], 4.05)
    assert state.actions == [Action.PAUSE]
    assert state.mode is Mode.PAUSED


def test_force_pause_and_resume(cfg: GestureConfig) -> None:
    engine = GestureEngine(cfg)
    unlock(engine)
    engine.force_pause()
    assert engine.mode is Mode.PAUSED
    paused = engine.update([make_hand("point")], 0.4)
    assert paused.actions == []
    assert paused.mode is Mode.PAUSED
    engine.force_resume()
    assert engine.mode is Mode.POINT
    state = engine.update([make_hand("point")], 0.5)
    assert state.mode is Mode.POINT
    assert state.cursor is not None


def test_primary_hand_prefers_right_and_switch_locks(cfg: GestureConfig) -> None:
    engine = GestureEngine(cfg)
    right = make_hand("point", label="Right", dx=200)
    left = make_hand("point", label="Left", dx=0)
    both = engine.update([left, right], 0.1)
    assert both.active_label == "Right"
    assert both.cursor is not None
    assert both.cursor[0] == pytest.approx(right.norm(6)[0])

    # Tangan kiri melakukan pinch saat tangan kanan aktif: diabaikan.
    ignored = engine.update([make_hand("point", pinch="index", label="Left"), right], 0.2)
    assert ignored.actions == []
    assert ignored.locked is False  # kanan netral membuka lock

    # Tangan kanan hilang; kiri (yang sedang pinch) mengambil alih tanpa klik.
    switched = engine.update(
        [make_hand("point", pinch="index", label="Left")], 0.3
    )
    assert switched.active_label == "Left"
    assert switched.actions == []
    assert switched.locked

    neutral = engine.update([left], 0.4)
    assert not neutral.locked
    click = engine.update([make_hand("point", pinch="index", label="Left")], 0.5)
    assert click.actions == [Action.LEFT_DOWN]


def test_reacquire_with_existing_pinch_does_not_click(cfg: GestureConfig) -> None:
    engine = GestureEngine(cfg)
    unlock(engine)
    engine.update([], 0.3)
    pinched = make_hand("point", pinch="index")
    state = engine.update([pinched], 0.4)
    assert state.actions == []
    assert state.freeze is Freeze.RELEASE
    assert state.locked
    held = engine.update([pinched], 0.6)
    assert held.actions == []
    engine.update([make_hand("point")], 0.7)
    click = engine.update([pinched], 0.8)
    assert click.actions == [Action.LEFT_DOWN]


def test_ambiguous_pinch_does_not_click(cfg: GestureConfig) -> None:
    engine = GestureEngine(cfg)
    unlock(engine)
    # Jempol tepat di antara ujung telunjuk dan tengah: jarak hampir sama.
    ambiguous = make_hand("open", thumb=(477, 277))
    state = engine.update([ambiguous], 0.3)
    assert state.actions == []
    assert state.ambiguous
    assert state.pinch is not None and not state.pinch.active

    # Pinch yang jelas (telunjuk saja) langsung bekerja.
    clear = engine.update([make_hand("open", pinch="index")], 0.5)
    assert clear.actions == [Action.LEFT_DOWN]
    assert not clear.ambiguous


def test_click_cooldown_blocks_immediate_second_click(cfg: GestureConfig) -> None:
    engine = GestureEngine(cfg)
    unlock(engine)
    engine.update([make_hand("point", pinch="index")], 0.3)
    released = engine.update([make_hand("point")], 0.4)
    assert released.actions == [Action.LEFT_UP]

    # Jari tengah mungkin ikut dekat setelah pinch kiri lepas; jangan
    # langsung menembak klik kanan.
    blocked = engine.update([make_hand("point", pinch="middle")], 0.45)
    assert blocked.actions == []
    assert blocked.pinch is not None and not blocked.pinch.active

    # Cooldown sudah habis, tetapi pinch yang ditahan saat cooldown harus
    # dilepas dulu supaya tidak menjadi klik tak sengaja.
    still = engine.update([make_hand("point", pinch="middle")], 0.6)
    assert still.actions == []
    assert still.pinch is not None and not still.pinch.active
    engine.update([make_hand("point")], 0.65)

    started = engine.update([make_hand("point", pinch="middle")], 0.7)
    assert started.actions == []
    assert started.pinch is not None and started.pinch.active
    clicked = engine.update([make_hand("point")], 0.8)
    assert clicked.actions == [Action.RIGHT_CLICK]


def test_strict_click_requires_stable_candidate(cfg: GestureConfig) -> None:
    strict = replace(cfg, pinch_stable_ms=50)
    engine = GestureEngine(strict)
    unlock(engine)
    first = engine.update([make_hand("point", pinch="index")], 0.3)
    assert first.actions == []
    assert first.pinch is not None and not first.pinch.active
    stable = engine.update([make_hand("point", pinch="index")], 0.36)
    assert stable.actions == [Action.LEFT_DOWN]

    flapping = GestureEngine(strict)
    unlock(flapping)
    flapping.update([make_hand("point", pinch="index")], 0.3)
    flapping.update([make_hand("point", pinch="middle")], 0.34)
    flapping.update([make_hand("point", pinch="index")], 0.38)
    after = flapping.update([make_hand("point", pinch="index")], 0.44)
    assert after.actions == [Action.LEFT_DOWN]


def test_scroll_works_even_when_clicks_locked(cfg: GestureConfig) -> None:
    engine = GestureEngine(cfg)
    first = engine.update([make_hand("scroll")], 0.1)
    assert first.locked  # tangan baru muncul: klik dikunci sementara
    state = engine.update([make_hand("scroll")], 0.25)
    assert state.actions == [Action.SCROLL_START]
    assert state.mode is Mode.SCROLL


def test_disabled_right_click_never_becomes_left(cfg: GestureConfig) -> None:
    engine = GestureEngine(replace(cfg, right_click=False))
    unlock(engine)
    state = engine.update([make_hand("point", pinch="middle")], 0.3)
    assert state.actions == []
    assert state.pinch is not None and not state.pinch.active
