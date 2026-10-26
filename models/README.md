# Model MediaPipe

`hand_landmarker.task` adalah model resmi **MediaPipe Hand Landmarker**
(Google) untuk Tasks API.

- Sumber:
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
- SHA-256:
  `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`
- Ukuran: 7.819.105 byte
- Lisensi: Apache License 2.0 (model MediaPipe)
- Dokumentasi:
  https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

Aplikasi memakai model ini secara lokal lewat `mediapipe.tasks.python.vision`.
Tidak ada gambar webcam yang dikirim ke internet. Jika model tidak ada,
`handtrack.hands.ensure_model` mengunduhnya sekali dan memverifikasi hash di
atas.

## Classifier gestur

`gesture_classifier.npz` **tidak disertakan**; file ini dihasilkan dari tangan
Anda sendiri lewat `collect_hand_data.py` + `train_gesture.py` (lihat README
utama, bagian 3). Isinya:

- `W1,b1,W2,b2,W3,b3`: bobot MLP 47 -> 128 -> 64 -> jumlah kelas. Untuk model
  ensemble, dimensi pertama adalah jumlah model, mis. `W1` (5,47,128),
  `b1` (5,128), `W3` (5,64,6). Model single lama tetap bisa dibaca.
- `mean,std`: standardisasi fitur dari data latih.
- `classes`: nama kelas (`neutral`, `left`, `right`, `middle`, `fist`,
  `scroll`).
- `ensemble`, `label_smoothing`, `format_version`: metadata training; loader
  NumPy hanya membutuhkan bobot dan kelas.

Inferensi dilakukan murni dengan NumPy di `handtrack/classifier.py`; PyTorch
hanya dipakai saat training. Data rekaman dan file model diabaikan Git.

