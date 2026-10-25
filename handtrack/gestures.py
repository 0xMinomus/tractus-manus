"""State machine gestur: pinch, tahan/drag, klik kanan/tengah, scroll, jeda."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum

from .classifier import Classifier
from .config import GestureConfig
from .hands import (
    CURSOR_LANDMARK,
    Hand,
    HandFeatures,
    Pinch,
    analyze_hand,
    select_active_hand,
)


class Mode(str, Enum):
    POINT = "POINT"
    SCROLL = "SCROLL"
    PAUSED = "PAUSED"


class Action(str, Enum):
    LEFT_DOWN = "left_down"
    LEFT_UP = "left_up"
    RIGHT_CLICK = "right_click"
    MIDDLE_CLICK = "middle_click"
    SCROLL_START = "scroll_start"
    SCROLL_END = "scroll_end"
    WHEEL_UP = "wheel_up"
    WHEEL_DOWN = "wheel_down"
    PAUSE = "pause"
    RESUME = "resume"


class Freeze(str, Enum):
    ANCHOR = "anchor"
    HOLD = "hold"
    RELEASE = "release"


@dataclass(frozen=True)
class PinchInfo:
    kind: Pinch
    distance: float
    active: bool


@dataclass
class FrameState:
    mode: Mode
    has_hand: bool
    cursor: tuple[float, float] | None
    actions: list[Action] = field(default_factory=list)
    freeze: Freeze | None = None
    active_label: str | None = None
    active_score: float | None = None
    pinch: PinchInfo | None = None
    pose: str | None = None
    pause_progress: float = 0.0
    scroll_progress: float = 0.0
    locked: bool = False
    ambiguous: bool = False
    ai_label: str | None = None
    ai_confidence: float = 0.0
    pinch_down: bool = False
    click_progress: float = 0.0


_PINCH_LABEL = {
    Pinch.INDEX: "left",
    Pinch.MIDDLE: "right",
    Pinch.RING: "middle",
}


class GestureEngine:
    """Menerjemahkan tangan aktif menjadi kursor + aksi mouse.

    Hanya satu tangan yang mengendalikan layar agar tidak ada dua kursor.
    Tangan kanan diutamakan bila kedua tangan terlihat.
    """

    def __init__(
        self,
        cfg: GestureConfig,
        primary_hand: str = "auto",
        classifier: Classifier | None = None,
        classifier_confidence: float = 0.65,
    ) -> None:
        self.cfg = cfg
        self.primary_hand = primary_hand
        self._classifier = classifier
        self._classifier_confidence = float(classifier_confidence)
        self._ai_label: str | None = None
        self._ai_conf: float = 0.0
        self.mode = Mode.POINT
        self._pinch: Pinch | None = None
        self._pinch_ai = False
        self._pinch_down = False
        self._pinch_at = 0.0
        self._pinch_start_distance = 0.0
        self._pending_pinch: Pinch | None = None
        self._pending_since: float | None = None
        self._last_label: str | None = None
        self._absent = True
        self._lock_gestures = False
        self._lock_until = 0.0

        self._pause_arm: float | None = None
        self._resume_arm: float | None = None
        self._pause_blocked = False
        self._click_cooldown_until = 0.0

        self._scroll_arm: float | None = None
        self._scroll_armed = True
        self._scroll_lost_at: float | None = None
        self._scroll_prev_y: float | None = None
        self._scroll_y: float | None = None
        self._scroll_accum = 0.0

    # ---------------------------------------------------------------- helpers
    @property
    def enabled_pinch(self) -> dict[Pinch, bool]:
        g = self.cfg
        return {
            Pinch.INDEX: g.left_click,
            Pinch.MIDDLE: g.right_click,
            Pinch.RING: g.middle_click,
        }

    def _ranked_pinch(self, features: HandFeatures) -> list[tuple[Pinch, float]]:
        return sorted(features.pinch.items(), key=lambda item: item[1])

    def _classifier_features(self, active: Hand, features: HandFeatures) -> HandFeatures:
        self._ai_label = None
        self._ai_conf = 0.0
        if self._classifier is None:
            return features
        label, conf = self._classifier.predict(active)
        self._ai_label = label
        self._ai_conf = float(conf)
        if conf < self._classifier_confidence:
            return features
        if label == "left":
            return replace(features, classifier_pinch=Pinch.INDEX, fist=False, scroll=False)
        if label == "right":
            return replace(features, classifier_pinch=Pinch.MIDDLE, fist=False, scroll=False)
        if label == "middle":
            return replace(features, classifier_pinch=Pinch.RING, fist=False, scroll=False)
        if label == "fist":
            return replace(features, fist=True, scroll=False, classifier_pinch=None)
        if label == "scroll":
            return replace(features, scroll=True, fist=False, classifier_pinch=None)
        # "neutral" dan label tak dikenal: aturan geometri tetap dipakai.
        return features

    def _ai_pinch_release(self, kind: Pinch, features: HandFeatures) -> bool:
        if self._ai_label != _PINCH_LABEL[kind]:
            return True
        if self._ai_conf < self._classifier_confidence:
            return True
        # AI boleh memilih pinch yang jaraknya di atas ambang rule. Lepas
        # ketika jari membuka dari jarak awal, bukan menunggu label AI yang
        # halus berubah (itu yang membuat klik singkat terasa jadi hold).
        distance = features.pinch.get(kind, float("inf"))
        threshold = max(
            self._release_for(kind),
            self._pinch_start_distance + 0.08,
        )
        return distance >= threshold

    def _press_for(self, kind: Pinch) -> float:
        return (
            self.cfg.middle_click_press
            if kind is Pinch.RING
            else self.cfg.pinch_press
        )

    def _release_for(self, kind: Pinch) -> float:
        return (
            self.cfg.middle_click_release
            if kind is Pinch.RING
            else self.cfg.pinch_release
        )

    def _pinch_candidate(self, features: HandFeatures) -> Pinch | None:
        """Pinch terdekat yang jelas menang dan melewati ambangnya.

        Jarak runner-up dijadikan pembanding: kalau jari lain hampir sama
        dekatnya, jangan menebak kiri/kanan. Ini mencegah klik tertukar.
        """
        if features.fist:
            return None
        if features.classifier_pinch is not None:
            kind = features.classifier_pinch
            return kind if self.enabled_pinch[kind] else None
        ranked = self._ranked_pinch(features)
        if not ranked:
            return None
        best, best_d = ranked[0]
        if not self.enabled_pinch[best]:
            return None
        if best_d > self._press_for(best):
            return None
        if len(ranked) > 1:
            second_d = ranked[1][1]
            if best_d > second_d * self.cfg.pinch_margin:
                return None
        return best

    def _ambiguous_pinch(self, features: HandFeatures) -> bool:
        """True bila pengguna jelas mencubit tetapi hasilnya ambigu/mati."""
        if features.fist:
            return False
        if features.classifier_pinch is not None:
            return False
        ranked = self._ranked_pinch(features)
        if not ranked:
            return False
        best, best_d = ranked[0]
        if not self.enabled_pinch[best] or best_d > self._press_for(best):
            return False
        return self._pinch_candidate(features) is None

    def _pinch_info(self, features: HandFeatures) -> PinchInfo | None:
        if self._pinch is not None:
            return PinchInfo(self._pinch, features.pinch.get(self._pinch, 0.0), True)
        if features.closest_pinch is not None:
            return PinchInfo(features.closest_pinch, features.closest_distance, False)
        return None

    def _release_button(self, click: bool, now: float) -> list[Action]:
        if self._pinch is None:
            return []
        kind = self._pinch
        was_down = self._pinch_down
        actions: list[Action] = []
        if kind is Pinch.INDEX:
            if was_down:
                actions.append(Action.LEFT_UP)
            elif click:
                # Tap: klik bersih di posisi anchor, tanpa fase hold.
                actions.extend((Action.LEFT_DOWN, Action.LEFT_UP))
        elif click:
            actions.append(
                Action.RIGHT_CLICK if kind is Pinch.MIDDLE else Action.MIDDLE_CLICK
            )
        self._pinch = None
        self._pinch_ai = False
        self._pinch_down = False
        if click and actions:
            self._click_cooldown_until = now + self.cfg.click_cooldown_ms / 1000.0
        return actions

    def _reset_pose_timers(self) -> None:
        self._pause_arm = None
        self._resume_arm = None
        self._scroll_arm = None
        self._scroll_lost_at = None
        self._scroll_prev_y = None
        self._scroll_y = None
        self._scroll_accum = 0.0
        self._pending_pinch = None
        self._pending_since = None
        self._pinch_ai = False
        self._pinch_down = False

    def _enter_scroll(self, active: Hand) -> None:
        self.mode = Mode.SCROLL
        self._scroll_arm = None
        self._scroll_armed = False
        self._scroll_lost_at = None
        self._scroll_accum = 0.0
        y = float(active.point(CURSOR_LANDMARK)[1])
        self._scroll_y = y
        self._scroll_prev_y = y

    def _exit_scroll(self, now: float) -> None:
        self.mode = Mode.POINT
        self._scroll_lost_at = None
        self._scroll_arm = None
        self._scroll_armed = False
        self._lock_gestures = True
        self._lock_until = now + 0.25

    def _lock_clicks(self, now: float) -> None:
        self._lock_gestures = True
        self._lock_until = now + 0.25

    def _state(
        self,
        *,
        active: Hand | None,
        features: HandFeatures | None,
        actions: list[Action],
        freeze: Freeze | None,
        cursor: tuple[float, float] | None = None,
        pause_progress: float = 0.0,
        scroll_progress: float = 0.0,
        ambiguous: bool = False,
        click_progress: float = 0.0,
    ) -> FrameState:
        pose = None
        if features is not None:
            if features.scroll:
                pose = "scroll"
            elif features.fist:
                pose = "fist"
        return FrameState(
            mode=self.mode,
            has_hand=active is not None,
            cursor=cursor,
            actions=actions,
            freeze=freeze,
            active_label=active.label if active else None,
            active_score=active.score if active else None,
            pinch=self._pinch_info(features) if features else None,
            pose=pose,
            pause_progress=max(0.0, min(1.0, pause_progress)),
            scroll_progress=max(0.0, min(1.0, scroll_progress)),
            locked=self._lock_gestures,
            ambiguous=ambiguous,
            ai_label=self._ai_label,
            ai_confidence=self._ai_conf,
            pinch_down=self._pinch is not None and self._pinch_down,
            click_progress=max(0.0, min(1.0, click_progress)),
        )

    def _unlock_if_neutral(self, features: HandFeatures, now: float) -> None:
        if not self._lock_gestures:
            return
        if self._pinch is not None or self._pinch_candidate(features) is not None:
            return
        # Hanya klik yang dikunci. Setelah 0,25 dtk (atau pose netral) klik
        # dibuka lagi; scroll dan kepalan tetap boleh dipakai selama lock.
        if now >= self._lock_until or (not features.scroll and not features.fist):
            self._lock_gestures = False

    # ------------------------------------------------------------------ update
    def update(self, hands: list[Hand], now: float) -> FrameState:
        actions: list[Action] = []
        freeze: Freeze | None = None
        active = select_active_hand(hands, self.primary_hand)

        # ---- tidak ada tangan aktif
        if active is None:
            if self._pinch is not None:
                actions.extend(self._release_button(click=False, now=now))
            if self.mode is Mode.SCROLL:
                actions.append(Action.SCROLL_END)
                self.mode = Mode.POINT
            if self.mode is Mode.PAUSED:
                self._pause_blocked = False
            self._last_label = None
            self._scroll_armed = True
            self._reset_pose_timers()
            self._ai_label = None
            self._ai_conf = 0.0
            if self._classifier is not None:
                self._classifier.reset()
            if not self._absent:
                self._absent = True
                freeze = Freeze.HOLD
            return self._state(
                active=None,
                features=None,
                actions=actions,
                freeze=freeze,
            )

        features = analyze_hand(active)

        # ---- tangan baru muncul / berganti
        if self._absent:
            self._absent = False
            self._lock_clicks(now)
            if self.mode is not Mode.PAUSED:
                freeze = Freeze.RELEASE
        if self._last_label is not None and active.label != self._last_label:
            actions.extend(self._release_button(click=False, now=now))
            if self.mode is Mode.SCROLL:
                actions.append(Action.SCROLL_END)
                self.mode = Mode.POINT
            self._reset_pose_timers()
            self._scroll_armed = True
            self._ai_label = None
            self._ai_conf = 0.0
            if self._classifier is not None:
                self._classifier.reset()
            self._lock_clicks(now)
            if self.mode is not Mode.PAUSED:
                freeze = Freeze.RELEASE
        self._last_label = active.label

        features = self._classifier_features(active, features)

        # ---- mode jeda: hanya kepalan tangan yang bisa melanjutkan
        if self.mode is Mode.PAUSED:
            progress = 0.0
            if features.fist and not self._pause_blocked:
                if self._resume_arm is None:
                    self._resume_arm = now
                hold_s = self.cfg.pause.hold_ms / 1000.0
                elapsed = now - self._resume_arm
                progress = elapsed / hold_s
                if elapsed >= hold_s:
                    self.mode = Mode.POINT
                    self._resume_arm = None
                    self._pause_arm = None
                    self._pause_blocked = True
                    self._lock_clicks(now)
                    actions.append(Action.RESUME)
                    freeze = Freeze.RELEASE
                    return self._state(
                        active=active,
                        features=features,
                        actions=actions,
                        freeze=freeze,
                        pause_progress=progress,
                    )
            else:
                self._resume_arm = None
                if not features.fist:
                    self._pause_blocked = False
            return self._state(
                active=active,
                features=features,
                actions=actions,
                freeze=freeze,
                pause_progress=progress,
            )

        # ---- mode scroll: kursor dibekukan, gerak vertikal = roda mouse
        if self.mode is Mode.SCROLL:
            progress = 0.0
            if self._pinch_candidate(features) is not None:
                actions.extend(self._release_button(click=False, now=now))
                actions.append(Action.SCROLL_END)
                self._exit_scroll(now)
                self._reset_pose_timers()
                return self._state(
                    active=active,
                    features=features,
                    actions=actions,
                    freeze=Freeze.RELEASE,
                )

            if not features.scroll:
                if self._scroll_lost_at is None:
                    self._scroll_lost_at = now
                lost = now - self._scroll_lost_at
                progress = lost / max(self.cfg.scroll.exit_hold_ms / 1000.0, 1e-6)
                # Jangan hasilkan wheel saat pose scroll sedang hilang.
                self._scroll_prev_y = None
                self._scroll_y = None
                if lost >= self.cfg.scroll.exit_hold_ms / 1000.0:
                    actions.append(Action.SCROLL_END)
                    self._exit_scroll(now)
                    self._reset_pose_timers()
                    return self._state(
                        active=active,
                        features=features,
                        actions=actions,
                        freeze=Freeze.RELEASE,
                    )
            else:
                self._scroll_lost_at = None
                y = float(active.point(CURSOR_LANDMARK)[1])
                alpha = max(min(self.cfg.scroll.smooth, 1.0), 0.05)
                if self._scroll_y is None:
                    self._scroll_y = y
                else:
                    self._scroll_y = alpha * y + (1.0 - alpha) * self._scroll_y
                if self._scroll_prev_y is None:
                    self._scroll_prev_y = self._scroll_y
                dy = self._scroll_prev_y - self._scroll_y
                self._scroll_prev_y = self._scroll_y
                self._scroll_accum += dy
                step = max(float(self.cfg.scroll.step_px), 1.0)
                while self._scroll_accum >= step:
                    actions.append(
                        Action.WHEEL_DOWN if self.cfg.scroll.invert else Action.WHEEL_UP
                    )
                    self._scroll_accum -= step
                while self._scroll_accum <= -step:
                    actions.append(
                        Action.WHEEL_UP if self.cfg.scroll.invert else Action.WHEEL_DOWN
                    )
                    self._scroll_accum += step

            return self._state(
                active=active,
                features=features,
                actions=actions,
                freeze=freeze,
                scroll_progress=progress,
            )

        # ---- mode normal: kursor + pinch + gerbang scroll/jeda
        self._unlock_if_neutral(features, now)

        # Pinch yang sedang aktif ditahan sampai dilepas (hysteresis).
        if self._pinch is not None:
            kind = self._pinch
            was_down = self._pinch_down
            if self._pinch_ai:
                release = self._ai_pinch_release(kind, features)
            else:
                distance = features.pinch.get(kind, float("inf"))
                release = distance >= self._release_for(kind)
            click_progress = 0.0
            if release:
                actions.extend(self._release_button(click=True, now=now))
                # Tap kiri tetap di anchor sebentar supaya dua klik cepat
                # mendarat di titik yang sama (double-click lebih mudah).
                freeze = (
                    Freeze.ANCHOR
                    if kind is Pinch.INDEX and not was_down
                    else Freeze.RELEASE
                )
                self._pause_arm = None
                self._scroll_arm = None
            elif kind is Pinch.INDEX and not self._pinch_down:
                hold_s = self.cfg.left_hold_ms / 1000.0
                elapsed = now - self._pinch_at
                if hold_s <= 0 or elapsed >= hold_s:
                    self._pinch_down = True
                    actions.append(Action.LEFT_DOWN)
                    freeze = Freeze.ANCHOR
                else:
                    click_progress = elapsed / hold_s
                    freeze = Freeze.HOLD
            cursor = active.norm(CURSOR_LANDMARK)
            return self._state(
                active=active,
                features=features,
                actions=actions,
                freeze=freeze,
                cursor=cursor,
                click_progress=click_progress,
            )

        kind = self._pinch_candidate(features)
        if kind is None:
            self._pending_pinch = None
            self._pending_since = None
        elif now < self._click_cooldown_until or self._lock_gestures:
            # Cooldown hanya menunda awal pinch baru; pinch yang masih
            # dipegang setelah cooldown boleh lanjut (penting untuk double-click).
            self._pending_pinch = None
            self._pending_since = None
        else:
            if kind is not self._pending_pinch:
                self._pending_pinch = kind
                self._pending_since = now
            stable_s = self.cfg.pinch_stable_ms / 1000.0
            if (
                self._pending_since is not None
                and now - self._pending_since >= stable_s
            ):
                self._pinch = kind
                self._pinch_ai = features.classifier_pinch is not None
                self._pinch_at = now
                self._pinch_start_distance = features.pinch.get(kind, 0.0)
                self._pending_pinch = None
                self._pending_since = None
                self._pause_arm = None
                self._scroll_arm = None
                if kind is Pinch.INDEX and self.cfg.left_hold_ms <= 0:
                    self._pinch_down = True
                    actions.append(Action.LEFT_DOWN)
                    freeze = Freeze.ANCHOR
                else:
                    self._pinch_down = False
                    freeze = Freeze.HOLD
                cursor = active.norm(CURSOR_LANDMARK)
                return self._state(
                    active=active,
                    features=features,
                    actions=actions,
                    freeze=freeze,
                    cursor=cursor,
                )

        # Jeda dengan kepalan tangan yang ditahan.
        pause_progress = 0.0
        if self.cfg.pause.enabled and features.fist:
            if not self._pause_blocked:
                if self._pause_arm is None:
                    self._pause_arm = now
                hold_s = self.cfg.pause.hold_ms / 1000.0
                pause_progress = (now - self._pause_arm) / hold_s
                if now - self._pause_arm >= hold_s:
                    self.mode = Mode.PAUSED
                    self._pause_arm = None
                    self._resume_arm = None
                    self._pause_blocked = True
                    self._scroll_arm = None
                    self._scroll_armed = True
                    actions.append(Action.PAUSE)
                    return self._state(
                        active=active,
                        features=features,
                        actions=actions,
                        freeze=Freeze.HOLD,
                        pause_progress=1.0,
                    )
        else:
            self._pause_arm = None
            if not features.fist:
                self._pause_blocked = False

        # Gerbang scroll: tunjukkan dua jari (telunjuk + tengah) selama sesaat.
        scroll_progress = 0.0
        if (
            self.cfg.scroll.enabled
            and now >= self._click_cooldown_until
        ):
            if features.scroll and self._scroll_armed:
                if self._scroll_arm is None:
                    self._scroll_arm = now
                hold_s = self.cfg.scroll.enter_hold_ms / 1000.0
                scroll_progress = (now - self._scroll_arm) / max(hold_s, 1e-6)
                if now - self._scroll_arm >= hold_s:
                    self._enter_scroll(active)
                    actions.append(Action.SCROLL_START)
                    return self._state(
                        active=active,
                        features=features,
                        actions=actions,
                        freeze=Freeze.HOLD,
                        pause_progress=pause_progress,
                        scroll_progress=1.0,
                    )
            else:
                self._scroll_arm = None
                if not features.scroll:
                    self._scroll_armed = True

        return self._state(
            active=active,
            features=features,
            actions=actions,
            freeze=freeze,
            cursor=active.norm(CURSOR_LANDMARK),
            pause_progress=pause_progress,
            scroll_progress=scroll_progress,
            ambiguous=self._ambiguous_pinch(features),
        )

    # ----------------------------------------------------------- kontrol luar
    def force_pause(self, now: float = 0.0) -> None:
        """Jeda dari keyboard; pemanggil wajib melepas semua tombol backend."""
        self.mode = Mode.PAUSED
        self._pinch = None
        self._pinch_ai = False
        self._pause_arm = None
        self._resume_arm = None
        self._pause_blocked = True
        self._scroll_arm = None
        self._scroll_armed = False
        self._reset_pose_timers()
        self._lock_clicks(now)

    def force_resume(self, now: float = 0.0) -> None:
        if self.mode is not Mode.PAUSED:
            return
        self.mode = Mode.POINT
        self._pause_arm = None
        self._resume_arm = None
        self._pause_blocked = True
        self._scroll_armed = False
        self._reset_pose_timers()
        self._lock_clicks(now)
