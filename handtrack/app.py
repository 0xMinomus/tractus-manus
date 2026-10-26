"""Loop aplikasi: kamera -> MediaPipe -> gestur -> mouse + HUD."""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, replace

import cv2

from .camera import open_camera, resize_for_processing
from .classifier import GestureClassifier, resolve_model_path
from .config import AppConfig
from .cursor import CursorController
from .gestures import Action, FrameState, Freeze, GestureEngine, Mode
from .hands import HandTracker
from .input import WHEEL_DELTA, MouseBackend, create_mouse
from .overlay import WINDOW_NAME, Overlay
from .winapi import is_ctrl_down, position_window, screen_bounds, set_always_on_top

log = logging.getLogger(__name__)


@dataclass
class RunOptions:
    dry_run: bool = False
    duration: float | None = None
    preview: bool | None = None
    debug: bool | None = None


class HandTrackApp:
    def __init__(self, cfg: AppConfig, options: RunOptions) -> None:
        ui = cfg.ui
        if options.preview is not None:
            ui = replace(ui, show_preview=options.preview)
        if options.debug is not None:
            ui = replace(ui, debug=options.debug)
        self.cfg = replace(cfg, ui=ui)
        self.options = options
        self._quit = False
        self._fps = 0.0
        self._last_frame_at: float | None = None
        self._frames = 0
        self._window_ready = False
        self._topmost = self.cfg.ui.always_on_top
        self._black_since: float | None = None
        self._black_warned = False
        self._low_fps_warned = False

    # ------------------------------------------------------------------ helpers
    def _print_banner(self, bounds) -> None:
        cfg = self.cfg
        print()
        print("=" * 62)
        print("  Tractus Manus - kontrol mouse dengan gestur tangan")
        print("=" * 62)
        print(f"  Kamera        : index {cfg.camera.index} "
              f"({cfg.camera.width}x{cfg.camera.height} diminta)")
        print(f"  Layar target  : {cfg.control.target_monitor} "
              f"{bounds.width}x{bounds.height} @ ({bounds.left},{bounds.top})")
        hand_desc = {
            "auto": "auto (kanan diutamakan)",
            "right": "kanan",
            "left": "kiri",
        }[cfg.control.primary_hand]
        print(f"  Tangan aktif  : {hand_desc}")
        print(f"  Sensitivitas  : {cfg.control.sensitivity:.1f} ( , / . = kurangi / tambah )")
        print(f"  Label tangan  : "
              f"{'ditukar' if cfg.camera.swap_handedness else 'normal'} (H = ubah)")
        if self.options.dry_run:
            print("  Mode          : LATIHAN - mouse TIDAK digerakkan")
        if self.options.duration:
            print(f"  Durasi        : {self.options.duration:.0f} detik (uji)")
        print("  Gestur        :")
        print("    JEPIT TELUNJUK + JEMPOL     -> nempel = klik, tahan hampir = drag")
        print("    JEPIT TENGAH + JEMPOL       -> klik kanan")
        print("    JEPIT JARI MANIS + JEMPOL   -> klik tengah (M = on/off)")
        print("    TELUNJUK + TENGAH (2 jari)  -> scroll (tahan 0,12 dtk)")
        print("    KEPAL TANGAN                -> jeda / lanjut (tahan 1 dtk)")
        print("  Kursor        : ikuti buku jari telunjuk (stabil saat pinch)")
        if self.cfg.ui.show_preview:
            print("  Preview       : Q/ESC keluar | P jeda | [ ] smoothing | R reset")
            print("                  , . sensitivitas | H label tangan | T topmost")
            print("                  M klik tengah on/off | D debug overlay")
        else:
            print("  Preview       : nonaktif (Ctrl+C untuk keluar)")
        ccfg = cfg.gestures.classifier
        status = f"aktif ({ccfg.model_path})" if ccfg.enabled else "nonaktif (opsional)"
        print(f"  AI gestur     : {status}")
        print("=" * 62)
        print()

    def _apply_action(self, action: Action, mouse: MouseBackend) -> None:
        if action is Action.LEFT_DOWN:
            mouse.left_down()
        elif action is Action.LEFT_UP:
            mouse.left_up()
        elif action is Action.RIGHT_CLICK:
            mouse.right_click()
        elif action is Action.MIDDLE_CLICK:
            mouse.middle_click()
        elif action is Action.WHEEL_UP:
            mouse.wheel(WHEEL_DELTA)
        elif action is Action.WHEEL_DOWN:
            mouse.wheel(-WHEEL_DELTA)
        elif action is Action.PAUSE:
            log.info("JEDA (gestur kepalan)")
        elif action is Action.RESUME:
            log.info("LANJUT")
        elif action is Action.SCROLL_START:
            log.info("Mode SCROLL")
        elif action is Action.SCROLL_END:
            log.info("Keluar SCROLL")
        if action in (
            Action.LEFT_DOWN,
            Action.LEFT_UP,
            Action.RIGHT_CLICK,
            Action.MIDDLE_CLICK,
        ):
            log.debug("Aksi mouse: %s", action.value)

    def _apply_state(
        self,
        state: FrameState,
        cursor: CursorController,
        mouse: MouseBackend,
        now: float,
    ) -> tuple[int, int] | None:
        """Terapkan aksi + freeze + kursor untuk satu frame (dipakai loop & test)."""
        for action in state.actions:
            self._apply_action(action, mouse)

        if state.freeze is Freeze.ANCHOR:
            cursor.freeze(now, self.cfg.control.click_anchor_ms / 1000.0)
        elif state.freeze is Freeze.HOLD:
            cursor.freeze(now, None)
        elif state.freeze is Freeze.RELEASE:
            cursor.unfreeze(now)

        if state.cursor is not None:
            return cursor.update(state.cursor, now)
        return None

    def _handle_key(
        self,
        key: int,
        now: float,
        engine: GestureEngine,
        cursor: CursorController,
        mouse: MouseBackend,
        overlay: Overlay,
        tracker: HandTracker,
    ) -> None:
        if key in (27, ord("q"), ord("Q")):
            self._quit = True
        elif key in (ord("p"), ord("P")):
            if engine.mode is Mode.PAUSED:
                engine.force_resume(now)
                cursor.unfreeze(now)
                overlay.flash("LANJUT", now)
                log.info("LANJUT (keyboard)")
            else:
                engine.force_pause(now)
                mouse.release_all()
                cursor.freeze(now, None)
                log.info("JEDA (keyboard)")
        elif key == ord("["):
            cursor.set_smoothing(cursor.smoothing - 0.2)
            overlay.flash(f"SMOOTHING {cursor.smoothing:.1f}", now, 1.2)
        elif key == ord("]"):
            cursor.set_smoothing(cursor.smoothing + 0.2)
            overlay.flash(f"SMOOTHING {cursor.smoothing:.1f}", now, 1.2)
        elif key in (ord(","), ord(".")):
            delta = 0.1 if key == ord(".") else -0.1
            value = cursor.set_sensitivity(cursor.sensitivity + delta)
            overlay.area = cursor.area
            overlay.sensitivity = value
            overlay.flash(f"SENSITIVITAS {value:.1f}", now, 1.0)
        elif key in (ord("r"), ord("R")):
            cursor.reset()
            overlay.flash("RESET", now, 0.8)
        elif key in (ord("d"), ord("D")):
            overlay.toggle_debug()
            overlay.flash(f"DEBUG {'ON' if overlay.debug else 'OFF'}", now, 0.8)
        elif key in (ord("m"), ord("M")) or (key == 13 and is_ctrl_down()):
            enabled = not engine.cfg.middle_click
            canceled = engine.set_middle_click(enabled)
            if canceled:
                cursor.unfreeze(now)
            overlay.flash(
                f"KLIK TENGAH {'ON' if enabled else 'OFF'}", now, 1.2
            )
            log.info("Klik tengah: %s", "aktif" if enabled else "nonaktif")
        elif key in (ord("h"), ord("H")):
            tracker.swap_handedness = not tracker.swap_handedness
            overlay.flash(
                "LABEL TANGAN DITUKAR"
                if tracker.swap_handedness
                else "LABEL TANGAN NORMAL",
                now,
                1.2,
            )
            log.info(
                "Label tangan: %s",
                "ditukar" if tracker.swap_handedness else "normal",
            )
        elif key in (ord("t"), ord("T")):
            self._topmost = not self._topmost
            set_always_on_top(WINDOW_NAME, self._topmost)
            overlay.flash(
                "PREVIEW SELALU DI ATAS" if self._topmost else "PREVIEW BIASA",
                now,
                1.0,
            )

    def _load_classifier(self) -> GestureClassifier | None:
        ccfg = self.cfg.gestures.classifier
        if not ccfg.enabled:
            return None
        path = resolve_model_path(ccfg.model_path)
        if not path.exists():
            log.warning(
                "AI gestur aktif tetapi model tidak ada: %s. "
                "Rekam data lalu jalankan train_gesture.py, atau matikan "
                "gestures.classifier.enabled.",
                path,
            )
            return None
        try:
            classifier = GestureClassifier(path, ccfg.smoothing)
        except Exception as exc:
            log.warning("Gagal memuat model AI %s: %s", path, exc)
            return None
        log.info(
            "AI gestur aktif: %s (%d kelas, smoothing %.2f)",
            path.name,
            len(classifier.classes),
            classifier.smoothing,
        )
        return classifier

    # --------------------------------------------------------------------- run
    def run(self) -> None:
        cfg = self.cfg
        bounds = screen_bounds(cfg.control.target_monitor)
        self._print_banner(bounds)
        classifier = self._load_classifier()

        cap = None
        tracker = None
        mouse = None
        try:
            cap = open_camera(cfg.camera)
            tracker = HandTracker(
                cfg.tracking,
                mirror=cfg.camera.mirror,
                swap_handedness=cfg.camera.swap_handedness,
            )
            mouse = create_mouse(self.options.dry_run, cfg.control.target_monitor)
            cursor = CursorController(bounds, cfg.control)
            engine = GestureEngine(
                cfg.gestures,
                cfg.control.primary_hand,
                classifier=classifier,
                classifier_confidence=cfg.gestures.classifier.confidence,
            )
            overlay = Overlay(
                cursor.area,
                cfg.ui,
                cfg.gestures.pinch_press,
                bounds,
                sensitivity=cursor.sensitivity,
            )
            if cfg.ui.show_preview:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

            started = time.perf_counter()
            last_mode = engine.mode
            last_status = 0.0
            read_failures = 0
            while not self._quit:
                frame_at = time.perf_counter()
                ok, frame = cap.read()
                if not ok or frame is None:
                    read_failures += 1
                    if read_failures == 1:
                        log.warning("Gagal membaca frame kamera; mencoba lagi...")
                    if read_failures > 90:
                        raise RuntimeError("Kamera berhenti mengirim frame.")
                    time.sleep(0.01)
                    continue
                read_failures = 0
                if float(frame.mean()) < 2.0:
                    if self._black_since is None:
                        self._black_since = frame_at
                    elif not self._black_warned and frame_at - self._black_since >= 3.0:
                        self._black_warned = True
                        log.warning(
                            "Frame kamera hitam total. Cek penutup privasi kamera, "
                            "izin kamera Windows, atau aplikasi lain yang memakai kamera."
                        )
                else:
                    self._black_since = None
                    self._black_warned = False
                if self._last_frame_at is not None:
                    dt = frame_at - self._last_frame_at
                    if 1e-4 < dt < 5.0:
                        inst = min(1.0 / dt, 240.0)
                        self._fps = inst if self._fps <= 0 else 0.9 * self._fps + 0.1 * inst
                self._last_frame_at = frame_at
                if (
                    not self._low_fps_warned
                    and 0.0 < self._fps < 5.0
                    and frame_at - started >= 5.0
                ):
                    self._low_fps_warned = True
                    log.warning(
                        "FPS kamera sangat rendah (%.1f). Tutup aplikasi lain yang "
                        "memakai kamera atau coba index kamera lain (--camera N).",
                        self._fps,
                    )

                frame = resize_for_processing(frame, cfg.tracking.process_width)
                frame, hands = tracker.process(frame)
                now = time.perf_counter()
                state = engine.update(hands, now)

                pos = self._apply_state(state, cursor, mouse, now)
                if pos is not None:
                    mouse.move(*pos)

                if state.mode is not last_mode:
                    log.info("Mode: %s -> %s", last_mode.value, state.mode.value)
                    last_mode = state.mode

                if not cfg.ui.show_preview:
                    if now - last_status >= 5.0:
                        last_status = now
                        hand = state.active_label or "-"
                        log.info(
                            "fps=%.1f mode=%s tangan=%s pinch=%s",
                            self._fps,
                            state.mode.value,
                            hand,
                            "-" if state.pinch is None else f"{state.pinch.distance:.2f}",
                        )
                    if self.options.duration and now - started >= self.options.duration:
                        log.info("Durasi uji selesai (%.1f detik).", self.options.duration)
                        break
                    continue

                overlay.area = cursor.area
                overlay.draw(
                    frame,
                    hands,
                    state,
                    now,
                    self._fps,
                    cursor.position,
                    cursor.smoothing,
                )
                cv2.imshow(WINDOW_NAME, frame)
                if not self._window_ready:
                    self._window_ready = True
                    position_window(
                        WINDOW_NAME,
                        max(bounds.left, bounds.right - frame.shape[1] - 24),
                        bounds.top + 24,
                    )
                    if self._topmost:
                        set_always_on_top(WINDOW_NAME, True)
                key = cv2.waitKey(1) & 0xFF
                if key != 0xFF:
                    self._handle_key(key, now, engine, cursor, mouse, overlay, tracker)
                self._frames += 1
                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        self._quit = True
                except cv2.error:
                    pass

                if self.options.duration and now - started >= self.options.duration:
                    log.info("Durasi uji selesai (%.1f detik).", self.options.duration)
                    break
        finally:
            if mouse is not None:
                mouse.release_all()
            if tracker is not None:
                tracker.close()
            if cap is not None:
                cap.release()
            if cfg.ui.show_preview:
                with contextlib.suppress(cv2.error):
                    cv2.destroyAllWindows()
        print("Tractus Manus selesai. Semua tombol mouse sudah dilepas.")
