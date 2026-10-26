"""HUD OpenCV: skeleton, status gestur, progress bar, preview layar."""

from __future__ import annotations

import cv2
import numpy as np

from .config import ActiveAreaConfig, UIConfig
from .gestures import Action, FrameState, Mode
from .hands import CURSOR_LANDMARK, HAND_CONNECTIONS, Hand, Pinch, analyze_hand
from .mapping import ScreenBounds

WINDOW_NAME = "HandTrack"
FONT = cv2.FONT_HERSHEY_SIMPLEX

WHITE = (238, 238, 238)
MUTED = (160, 160, 160)
PANEL = (18, 18, 18)
GREEN = (112, 232, 144)
AMBER = (72, 200, 255)
RED = (86, 86, 240)
BLUE = (246, 176, 86)

_FLASH_TEXT = {
    Action.LEFT_DOWN: "KLIK KIRI DITAHAN",
    Action.LEFT_UP: "KLIK KIRI",
    Action.RIGHT_CLICK: "KLIK KANAN",
    Action.MIDDLE_CLICK: "KLIK TENGAH",
    Action.SCROLL_START: "MODE SCROLL",
    Action.WHEEL_UP: "SCROLL",
    Action.WHEEL_DOWN: "SCROLL",
    Action.RESUME: "LANJUT",
}


def _alpha_rect(
    frame: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.6
) -> None:
    x = max(0, x)
    y = max(0, y)
    w = min(w, frame.shape[1] - x)
    h = min(h, frame.shape[0] - y)
    if w <= 0 or h <= 0:
        return
    roi = frame[y : y + h, x : x + w]
    overlay = np.empty_like(roi)
    overlay[:] = PANEL
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, roi)


def _text(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float = 0.5,
    color: tuple[int, int, int] = WHITE,
    thickness: int = 1,
) -> None:
    cv2.putText(
        frame, text, (x + 1, y + 1), FONT, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA
    )
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def _fit_scale(text: str, max_width: int, scale: float) -> float:
    (w, _), _ = cv2.getTextSize(text, FONT, scale, 1)
    if w <= max_width:
        return scale
    return max(0.28, scale * max_width / max(1, w))


def _bar(
    frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    fraction: float,
    color: tuple[int, int, int],
) -> None:
    fraction = max(0.0, min(1.0, fraction))
    cv2.rectangle(frame, (x, y), (x + width, y + height), (64, 64, 64), -1)
    if fraction > 0:
        cv2.rectangle(
            frame, (x, y), (x + int(width * fraction), y + height), color, -1
        )


def _corner_box(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    length: int = 22,
) -> None:
    for px, py, dx, dy in (
        (x1, y1, 1, 1),
        (x2, y1, -1, 1),
        (x1, y2, 1, -1),
        (x2, y2, -1, -1),
    ):
        cv2.line(frame, (px, py), (px + dx * length, py), color, 1, cv2.LINE_AA)
        cv2.line(frame, (px, py), (px, py + dy * length), color, 1, cv2.LINE_AA)


class Overlay:
    def __init__(
        self,
        area: ActiveAreaConfig,
        ui: UIConfig,
        pinch_press: float,
        screen: ScreenBounds,
        sensitivity: float = 1.0,
    ) -> None:
        self.area = area
        self.ui = ui
        self.sensitivity = sensitivity
        self.debug = ui.debug
        self.pinch_press = pinch_press
        self.screen = screen
        self._flash_text = ""
        self._flash_until = 0.0

    def toggle_debug(self) -> None:
        self.debug = not self.debug

    def flash(self, text: str, now: float, duration: float = 0.8) -> None:
        self._flash_text = text
        self._flash_until = now + duration

    # --------------------------------------------------------------- internals
    def _draw_hand(
        self,
        frame: np.ndarray,
        hand: Hand,
        active: bool,
        show_skeleton: bool,
    ) -> None:
        color = GREEN if active else MUTED
        if show_skeleton:
            for a, b in HAND_CONNECTIONS:
                pa = tuple(int(v) for v in hand.point(a))
                pb = tuple(int(v) for v in hand.point(b))
                cv2.line(frame, pa, pb, color, 2 if active else 1, cv2.LINE_AA)
            for index in range(21):
                point = tuple(int(v) for v in hand.point(index))
                cv2.circle(frame, point, 3 if active else 2, color, -1, cv2.LINE_AA)

        label = "KANAN" if hand.label == "Right" else "KIRI"
        wrist = tuple(int(v) for v in hand.point(0))
        x = max(6, min(wrist[0] - 24, frame.shape[1] - 70))
        y = max(18, wrist[1] - 18)
        _text(frame, label, x, y, 0.48, color, 1)

    def _draw_mini(
        self,
        frame: np.ndarray,
        cursor_px: tuple[int, int] | None,
    ) -> None:
        frame_w = frame.shape[1]
        if frame_w < 780:
            return
        mini_w = 196
        aspect = self.screen.height / max(1, self.screen.width)
        mini_h = max(70, int(mini_w * aspect))
        x = frame_w - mini_w - 14
        y = 14
        _alpha_rect(frame, x - 1, y - 1, mini_w + 2, mini_h + 24, 0.65)
        cv2.rectangle(frame, (x, y), (x + mini_w, y + mini_h), (80, 80, 80), 1)
        _text(frame, "PREVIEW LAYAR", x + 4, y + mini_h + 16, 0.36, MUTED, 1)

        if cursor_px is None:
            _text(
                frame,
                "TUNJUKKAN TANGAN",
                x + 14,
                y + mini_h // 2 + 4,
                0.4,
                MUTED,
                1,
            )
            return
        fx = (cursor_px[0] - self.screen.left) / max(1, self.screen.width - 1)
        fy = (cursor_px[1] - self.screen.top) / max(1, self.screen.height - 1)
        px = x + int(fx * (mini_w - 1))
        py = y + int(fy * (mini_h - 1))
        cv2.drawMarker(
            frame, (px, py), AMBER, cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA
        )

    def _draw_hud(
        self,
        frame: np.ndarray,
        state: FrameState,
        fps: float,
        smoothing: float,
        active_hand: Hand | None,
    ) -> None:
        frame_w = frame.shape[1]
        panel_w = min(500, frame_w - 28)
        panel_h = 136
        _alpha_rect(frame, 14, 14, panel_w, panel_h, 0.62)
        mode_color = (
            RED if state.mode is Mode.PAUSED else AMBER if state.mode is Mode.SCROLL else GREEN
        )
        mode_text = "JEDA" if state.mode is Mode.PAUSED else state.mode.value

        _text(frame, "HandTrack", 28, 46, 0.78, WHITE, 2)
        fps_text = f"{fps:4.0f} fps"
        (fw, _), _ = cv2.getTextSize(fps_text, FONT, 0.46, 1)
        _text(frame, fps_text, 14 + panel_w - fw - 14, 44, 0.46, MUTED, 1)

        _text(frame, f"MODE  {mode_text}", 28, 76, 0.56, mode_color, 1)

        if state.has_hand:
            side = "KANAN" if state.active_label == "Right" else "KIRI"
            score = int(round((state.active_score or 0.0) * 100))
            hand_text = f"TANGAN  {side}  {score}%"
        else:
            hand_text = "TANGAN  -"
        _text(frame, hand_text, 28, 102, 0.52, WHITE, 1)

        if not state.has_hand:
            detail, detail_color = "TUNJUKKAN TANGAN KE KAMERA", MUTED
        elif state.mode is Mode.PAUSED:
            # Layar jeda di tengah sudah memuat instruksi + progress.
            detail, detail_color = "", RED
        elif state.mode is Mode.SCROLL:
            detail, detail_color = "GESER TANGAN NAIK / TURUN", AMBER
        elif state.locked:
            detail, detail_color = "KLIK DIKUNCI - TUNGGU SEBENTAR", AMBER
        elif state.pinch is not None and state.pinch.active:
            if state.pinch.kind is Pinch.INDEX:
                if state.pinch_down:
                    detail = "KLIK KIRI DITAHAN - GESER UNTUK DRAG"
                elif state.touch_clicked:
                    detail = "KLIK TERKIRIM - TUTUP LAGI UNTUK DOUBLE"
                else:
                    detail = "NEMPEL = KLIK  |  TAHAN HAMPIR = DRAG"
            elif state.pinch.kind is Pinch.MIDDLE:
                detail = "LEPAS UNTUK KLIK KANAN"
            else:
                detail = "LEPAS UNTUK KLIK TENGAH"
            detail_color = GREEN
        elif state.ambiguous:
            detail, detail_color = "JEPIT AMBIGU - JAUHKAN JARI LAIN", AMBER
        elif state.pose == "fist":
            detail, detail_color = "KEPAL - TAHAN UNTUK JEDA", RED
        elif state.pose == "scroll":
            detail, detail_color = "2 JARI - TAHAN UNTUK SCROLL", AMBER
        else:
            detail, detail_color = "SIAP", MUTED
        _text(
            frame,
            detail,
            28,
            126,
            _fit_scale(detail, panel_w - 30, 0.48),
            detail_color,
            1,
        )

        progress = 0.0
        progress_color = AMBER
        if state.mode is Mode.SCROLL:
            progress = state.scroll_progress
        elif state.mode is Mode.PAUSED:
            # Progress jeda digambar besar di tengah layar.
            progress = 0.0
        elif state.pause_progress > 0:
            progress = state.pause_progress
            progress_color = RED
        elif state.click_progress > 0:
            progress = state.click_progress
            progress_color = GREEN
        elif state.scroll_progress > 0:
            progress = state.scroll_progress
            progress_color = AMBER
        if progress > 0:
            _bar(frame, 28, 133, panel_w - 28, 4, progress, progress_color)

        if self.debug:
            self._draw_debug(frame, state, smoothing, panel_w, active_hand)

    def _draw_debug(
        self,
        frame: np.ndarray,
        state: FrameState,
        smoothing: float,
        panel_w: int,
        active_hand: Hand | None,
    ) -> None:
        lines = [
            f"sm={smoothing:.1f}  sens={self.sensitivity:.1f}  "
            f"press={self.pinch_press:.2f}  "
            f"lock={'Y' if state.locked else 'N'}  "
            f"amb={'Y' if state.ambiguous else 'N'}"
        ]
        if state.ai_label is not None:
            lines.append(f"ai={state.ai_label} {state.ai_confidence:.2f}")
        if active_hand is not None:
            features = analyze_hand(active_hand)
            lines.append(
                "jepit "
                f"telunjuk={features.pinch[Pinch.INDEX]:.2f} "
                f"tengah={features.pinch[Pinch.MIDDLE]:.2f} "
                f"manis={features.pinch[Pinch.RING]:.2f}"
            )
            lines.append(
                "sudut "
                f"telunjuk={features.angles['index']:.0f} "
                f"tengah={features.angles['middle']:.0f} "
                f"manis={features.angles['ring']:.0f}"
            )
        if state.pinch is not None:
            lines.append(
                f"jepit {state.pinch.kind.name.lower()} {state.pinch.distance:.2f}"
            )
        y = 172
        text_w = max(cv2.getTextSize(line, FONT, 0.4, 1)[0][0] for line in lines)
        dbg_w = min(panel_w, text_w + 24)
        _alpha_rect(frame, 14, y - 16, dbg_w, 18 * len(lines) + 12, 0.55)
        for line in lines:
            _text(frame, line, 26, y, 0.4, MUTED, 1)
            y += 18

    def _draw_hints(self, frame: np.ndarray) -> None:
        frame_h, frame_w = frame.shape[:2]
        line1 = "NEMPELKAN JARI = KLIK   |   TAHAN HAMPIR = DRAG"
        line2 = "JEMPOL+TENGAH = KLIK KANAN | 2 JARI = SCROLL | KEPAL = JEDA"
        scale = _fit_scale(line1, frame_w - 52, 0.46)
        panel_w = min(
            max(
                cv2.getTextSize(line1, FONT, scale, 1)[0][0],
                cv2.getTextSize(line2, FONT, scale, 1)[0][0],
            )
            + 28,
            frame_w - 28,
        )
        y = frame_h - 62
        _alpha_rect(frame, 14, y, panel_w, 50, 0.58)
        _text(frame, line1, 26, y + 21, scale, WHITE, 1)
        _text(frame, line2, 26, y + 41, scale, WHITE, 1)

    def _draw_pause(self, frame: np.ndarray, state: FrameState) -> None:
        if state.mode is not Mode.PAUSED:
            return
        frame_h, frame_w = frame.shape[:2]
        dark = np.zeros_like(frame)
        cv2.addWeighted(frame, 0.38, dark, 0.62, 0, frame)
        title = "JEDA"
        (tw, th), _ = cv2.getTextSize(title, FONT, 1.5, 3)
        _text(frame, title, (frame_w - tw) // 2, frame_h // 2 - 6, 1.5, RED, 3)
        sub = "KEPAL TANGAN ATAU TEKAN P UNTUK LANJUT"
        (sw, _), _ = cv2.getTextSize(sub, FONT, 0.55, 1)
        _text(frame, sub, (frame_w - sw) // 2, frame_h // 2 + 30, 0.55, WHITE, 1)
        if state.pause_progress > 0:
            bar_w = min(360, frame_w - 80)
            _bar(
                frame,
                (frame_w - bar_w) // 2,
                frame_h // 2 + 48,
                bar_w,
                6,
                state.pause_progress,
                RED,
            )

    def _draw_flash(self, frame: np.ndarray, now: float) -> None:
        if not self._flash_text or now >= self._flash_until:
            return
        frame_h, frame_w = frame.shape[:2]
        (tw, th), _ = cv2.getTextSize(self._flash_text, FONT, 0.85, 2)
        x = (frame_w - tw) // 2
        y = int(frame_h * 0.36)
        _alpha_rect(frame, x - 18, y - th - 12, tw + 36, th + 24, 0.72)
        _text(frame, self._flash_text, x, y, 0.85, AMBER, 2)

    # ------------------------------------------------------------------- public
    def draw(
        self,
        frame: np.ndarray,
        hands: list[Hand],
        state: FrameState,
        now: float,
        fps: float,
        cursor_px: tuple[int, int] | None,
        smoothing: float,
    ) -> None:
        for action in state.actions:
            text = _FLASH_TEXT.get(action)
            if text:
                self.flash(text, now)

        frame_h, frame_w = frame.shape[:2]
        x1 = int(self.area.x_min * frame_w)
        x2 = int(self.area.x_max * frame_w)
        y1 = int(self.area.y_min * frame_h)
        y2 = int(self.area.y_max * frame_h)
        _corner_box(frame, x1, y1, x2, y2, (95, 95, 95), 24)

        active_index = None
        if state.active_label and hands:
            candidates = [
                i for i, hand in enumerate(hands) if hand.label == state.active_label
            ]
            if candidates:
                active_index = max(candidates, key=lambda i: hands[i].score)
        for index, hand in enumerate(hands):
            self._draw_hand(frame, hand, index == active_index, self.ui.show_skeleton)

        active_hand = hands[active_index] if active_index is not None else None
        if active_hand is not None:
            point = tuple(int(v) for v in active_hand.point(CURSOR_LANDMARK))
            cv2.circle(frame, point, 11, GREEN, 1, cv2.LINE_AA)
            cv2.circle(frame, point, 3, GREEN, -1, cv2.LINE_AA)

        self._draw_pause(frame, state)
        self._draw_hud(frame, state, fps, smoothing, active_hand)
        self._draw_mini(frame, cursor_px)
        self._draw_hints(frame)
        self._draw_flash(frame, now)
