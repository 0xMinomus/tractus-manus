"""Pembukaan kamera dan resize frame yang dipakai app maupun collector data."""

from __future__ import annotations

import logging
import time

import cv2

from .config import CameraConfig

log = logging.getLogger(__name__)

CAMERA_BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "any": cv2.CAP_ANY,
}


def open_camera(cam: CameraConfig) -> cv2.VideoCapture:
    backend = CAMERA_BACKENDS[cam.backend]
    cap = cv2.VideoCapture(cam.index, backend)
    if not cap.isOpened():
        raise RuntimeError(
            f"Kamera index {cam.index} tidak bisa dibuka. "
            "Tutup aplikasi lain yang memakai kamera, cek izin kamera Windows, "
            "lalu coba --camera 1."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.height)
    cap.set(cv2.CAP_PROP_FPS, cam.fps)
    # Ambil frame terbaru, bukan antrean lama (kalau backend mendukung).
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ok, frame = False, None
    for _ in range(20):
        ok, frame = cap.read()
        if ok:
            break
        time.sleep(0.05)
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(
            "Kamera terbuka tetapi frame tidak bisa dibaca. "
            "Coba backend lain di config.json (camera.backend: dshow/msmf)."
        )
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or frame.shape[1])
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or frame.shape[0])
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    log.info("Kamera aktif: %dx%d @ %.0f fps", actual_w, actual_h, actual_fps)
    return cap


def resize_for_processing(frame, process_width: int):
    if process_width <= 0 or frame.shape[1] <= process_width:
        return frame
    scale = process_width / frame.shape[1]
    return cv2.resize(
        frame,
        (process_width, max(1, round(frame.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
