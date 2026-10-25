# HandTrack

Kontrol mouse Windows sepenuhnya dengan gestur tangan dari webcam. Cukup satu
tangan; webcam tetap melacak dua tangan agar Anda bisa berpindah tangan tanpa
memutus kontrol.

- **Kursor** mengikuti buku jari telunjuk, dipetakan ke layar dengan filter
  One Euro yang dibuat lebih responsif (halus saat diam, cepat saat bergerak).
- **Klik kiri / tahan / drag** dengan pinch jempol + telunjuk. Tahan pinch =
  tombol kiri terus ditekan; geser tangan = drag.
- **Klik kanan** dengan pinch jempol + jari tengah.
- **Klik tengah** (opsional) dengan pinch rapat jempol + jari manis.
- **Scroll** dengan pose dua jari (telunjuk + tengah) lalu gerakkan tangan
  naik/turun.
- **Jeda / lanjut** dengan mengepalkan tangan dan menahannya sebentar, atau
  tombol `P`.
- Klik ganda: lakukan pinch dua kali cepat (Windows mendeteksinya sebagai
  double-click).
- Jendela preview menampilkan skeleton tangan, mode, progress bar, dan preview
  posisi kursor di layar.

Program ini **Windows-only** karena memakai `SendInput` untuk kontrol mouse.

Aturan gestur di atas bekerja tanpa model tambahan. Jika pinch Anda sering
ambigu (klik kiri/kanan/tengah tertukar, pose V sulit dibedakan), aplikasi juga
bisa memakai **classifier gestur yang dilatih dari tangan Anda sendiri**.
Lihat [bagian 3](#3-latih-classifier-gestur-opsional-disarankan).

## 1. Instalasi

Kebutuhan:

- Windows 10/11 64-bit
- Webcam yang berfungsi
- [uv](https://docs.astral.sh/uv/) (Python 3.12 akan disiapkan otomatis)

Langkah:

1. Jalankan `setup.bat` (sekali saja). Script ini menginstal Python 3.12 dan
   dependensi ke folder `.venv`.
2. Jalankan `run.bat`.

Manual, jika tidak ingin memakai script:

```bat
uv python install 3.12
uv sync
.venv\Scripts\python.exe main.py
```

Model MediaPipe sudah disertakan di `models/hand_landmarker.task`. Jika file itu
hilang, aplikasi akan mengunduhnya sekali dari server resmi MediaPipe dan
memverifikasi SHA-256 sebelum dipakai.

Untuk **melatih classifier gestur** (bagian 3), PyTorch dibutuhkan sekali.
Build CUDA untuk RTX 5060:

```bat
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Tanpa GPU, gunakan `uv sync --group train` (PyTorch CPU). Pelatihan tetap bisa
berjalan, hanya lebih lambat. Inferensi classifier memakai NumPy dan tidak
membutuhkan PyTorch.

Coba dulu tanpa menggerakkan mouse:

```bat
run.bat --dry-run
```

Semua aksi tercatat di terminal, kursor asli tidak disentuh.

## 2. Cara pakai

Posisikan kamera setinggi mata, jarak sekitar 50-80 cm, dengan pencahayaan dari
depan (bukan dari belakang). Tunjukkan satu tangan ke kamera. Tangan kanan
diutamakan; jika tangan kanan hilang, tangan kiri otomatis mengambil alih.

Satu hal penting: **kursor mengikuti buku jari telunjuk (PIP), bukan ujung
jari**. Titik ini stabil saat Anda melakukan pinch sehingga klik tidak bergeser.
Lingkaran hijau di preview menandai titik kursor yang sebenarnya.

### Gestur

| Gestur | Aksi |
| --- | --- |
| Gerakkan tangan (telunjuk) | Gerakkan kursor |
| Pinch jempol + telunjuk | Tekan klik kiri; tahan untuk hold/drag; lepas untuk klik |
| Pinch dua kali cepat | Double-click |
| Pinch jempol + jari tengah | Klik kanan (lepas pinch untuk mengeksekusi) |
| Pinch rapat jempol + jari manis | Klik tengah (opsional) |
| Pose V (telunjuk + tengah), tahan ±0,12 dtk | Masuk mode scroll; gerakkan tangan naik/turun |
| Kepal tangan, tahan ±1 dtk | Jeda; ulangi untuk lanjut |
| Telapak terbuka | Netral / tidak melakukan aksi |
| Kepalan (sebentar) | Tidak mengklik (mencegah klik tak sengaja) |

Catatan:

- Jeda tidak lagi memakai telapak terbuka—telapak sekarang netral. Pakai
  kepalan yang ditahan sekitar 1 detik.
- Saat pinch jempol + jari tengah, jaga agar ujung telunjuk tidak ikut
  menyentuh jempol. Aplikasi memilih jari yang **jelas paling dekat**; kalau
  jarak telunjuk dan tengah hampir sama, klik sengaja tidak dijalankan dan HUD
  menampilkan `JEPIT AMBIGU`. Setelah setiap klik ada jeda 150 ms, dan pinch
  yang ditahan saat jeda harus dilepas dulu sebelum boleh klik lagi.
- Klik memakai tiga aturan ketat: jari pemenang harus unik (margin 0,75),
  posisi pinch harus stabil ±25 ms, dan tidak ada klik selama cooldown 150 ms.
- Setelah tangan muncul atau berganti, hanya **klik** yang dikunci ±0,25 dtk;
  scroll dan kepalan tetap bisa dipakai. HUD menampilkan `KLIK DIKUNCI`.
- Jika label KANAN/KIRI terbalik, tekan `H` (atau ubah
  `camera.swap_handedness`).
- Jika sulit menjangkau sudut layar, tekan `.` untuk menaikkan sensitivitas
  (mis. 1,4-1,6); kurangi dengan `,`. Kotak area di preview ikut mengecil.
- Klik tengah memakai ambang pinch yang lebih ketat karena pada pose V jempol
  sering menyentuh jari manis yang terlipat. Jika tidak nyaman, matikan lewat
  `gestures.middle_click: false`.
- Pose V lebih toleran: cukup salah satu dari jari manis/kelingking
  terlipat, jadi kelingking boleh sedikit keluar. Telapak terbuka (keduanya
  terentang) tidak dianggap scroll. Tidak perlu menahan lama.
- Jendela preview selalu di atas secara default. Tekan `T` untuk
  mengubahnya; pindahkan jendela dengan drag seperti jendela Windows biasa.

### Tombol keyboard (saat preview fokus)

| Tombol | Fungsi |
| --- | --- |
| `Q` / `Esc` | Keluar (semua tombol mouse dilepas) |
| `P` | Jeda / lanjut |
| `[` / `]` | Kurangi / tambah smoothing (min cutoff 0,3-8,0) |
| `,` / `.` | Kurangi / tambah sensitivitas (0,5-3,0) |
| `H` | Tukar label tangan kiri/kanan |
| `R` | Reset filter dan posisi kursor |
| `D` | Tampilkan angka debug (jarak pinch, sudut, lock, sensitivitas) |
| `T` | Toggle preview selalu di atas |

Tanpa preview (`--no-preview`), gunakan `Ctrl+C` di terminal untuk keluar.

## 3. Latih classifier gestur (opsional, disarankan)

Aturan geometri sudah cukup untuk mulai. Classifier membantu saat jarak pinch
ambigu—misalnya klik tengah vs pose V, atau telunjuk dan tengah hampir sama
dekat ke jempol. Classifier **hanya memilih aksi** (klik kiri/kanan/tengah,
kepal/jeda, scroll, atau netral). Kursor tetap mengikuti landmark; tidak ada
model baru untuk kursor.

Model default **tidak disertakan**. Anda merekam beberapa menit data tangan
Anda sendiri, lalu melatih MLP kecil di GPU (RTX 5060). Data dan model tetap
lokal; tidak ada yang dikirim ke internet.

### 3.1 Rekam data

```bat
collect.bat
```

Jendela perekam menampilkan skeleton dan status. Tunjukkan **satu tangan**
(tangan mana pun; fitur otomatis menormalkan kiri/kanan).

| Tombol | Fungsi |
| --- | --- |
| `1` | label `neutral` |
| `2` | label `left` (pinch jempol + telunjuk) |
| `3` | label `right` (pinch jempol + tengah) |
| `4` | label `middle` (pinch rapat jempol + jari manis) |
| `5` | label `fist` (kepalan untuk jeda) |
| `6` | label `scroll` (pose V: telunjuk + tengah terbuka) |
| `Space` | mulai / berhenti merekam |
| `N` | sesi baru |
| `Q` / `Esc` | keluar |

Ganti label otomatis menjeda rekaman supaya data tidak tercampur. Rekam
masing-masing label **20-40 detik**, dengan variasi jarak tangan, posisi di
frame, sudut, dan pencahayaan. Lakukan **minimal 2 sesi** (`N`), dan isi
**semua 6 label di setiap sesi**—trainer memakai sesi sebagai pembatas
validasi, sehingga angka akurasinya jujur.

Pose yang direkam:

- `neutral`: tangan santai/terbuka atau menunjuk seperti saat menggerakkan
  kursor; bukan pinch, bukan kepalan, bukan pose V.
- `left` / `right` / `middle`: pinch yang Anda maksud untuk tiap klik.
- `fist`: kepalan tangan penuh.
- `scroll`: telunjuk + tengah terentang, jari manis/kelingking terlipat.

Data ditulis ke `data/gestures.jsonl` (satu baris per frame, berisi 21
landmark dan label). File ini privat dan diabaikan Git. Jangan mengubah
`camera.swap_handedness` antara perekaman dan pemakaian; label tangan ikut
disimpan untuk menormalkan fitur.

### 3.2 Latih model

```bat
train.bat
```

Trainer memakai GPU jika tersedia, memisahkan satu sesi untuk validasi,
menyeimbangkan kelas, dan menyimpan `models/gesture_classifier.npz`. Contoh
keluaran:

```
Akurasi train: 98.7%
Akurasi val  : 93.4%
```

Jika akurasi validasi rendah (di bawah ±85%), tambah sesi/data, pastikan pose
konsisten, dan jangan mencampur dua tangan dalam satu frame. Sesi validasi
yang tidak berisi semua label membuat confusion matrix sulit dibaca.

Opsi lain:

```bat
train.bat --data data/gestures.jsonl --out models/gesture_classifier.npz --epochs 200
```

### 3.3 Aktifkan classifier

Ubah `config.json`:

```json
"gestures": {
  "classifier": {
    "enabled": true,
    "model_path": "models/gesture_classifier.npz",
    "confidence": 0.65,
    "smoothing": 0.6
  }
}
```

Jalankan ulang `run.bat`. Banner akan menampilkan `AI gestur: aktif`; overlay
debug (`D`) menampilkan `ai=label confidence`. Jika model tidak ada, aplikasi
memberi peringatan dan tetap memakai aturan geometri.

Cara kerja saat aktif:

- Prediksi dengan confidence ≥ `confidence` menggantikan pilihan pinch/fist/
  scroll. Label `neutral` berarti kembali ke aturan geometri.
- Pinch yang dimulai oleh AI dilepas saat label AI berubah; ada pengaman
  geometri agar tombol tidak tertahan saat tangan jelas membuka. Kehilangan
  tangan tetap melepas semua tombol.
- `smoothing` 0-1: makin tinggi makin halus tetapi makin lambat bereaksi.
  Naikkan `confidence` jika masih salah; turunkan ke 0,55 jika gesture sering
  tidak terdeteksi.

Untuk mematikan lagi: `"enabled": false`.

## 4. Konfigurasi

Semua pengaturan ada di `config.json` (dibuat ulang dengan
`--write-config` jika belum ada). Perubahan dibaca saat aplikasi dijalankan.

| Key | Default | Arti |
| --- | --- | --- |
| `camera.index` | `0` | Index webcam; coba `1` jika kamera lain terbuka |
| `camera.width/height/fps` | `1280/720/60` | Resolusi & FPS yang diminta ke kamera |
| `camera.backend` | `dshow` | `dshow`, `msmf`, atau `any` |
| `camera.mirror` | `true` | Tampilan cermin (disarankan) |
| `camera.swap_handedness` | `true` | Tukar label kiri/kanan jika MediaPipe terbalik (bisa `H`) |
| `tracking.process_width` | `800` | Lebar frame untuk MediaPipe; turunkan ke `640`/`720` untuk lebih cepat, naikkan ke `960` untuk preview lebih besar |
| `tracking.max_hands` | `2` | 1 atau 2 |
| `tracking.min_*_confidence` | `0,6/0,55/0,55` | Ambang deteksi/tracking MediaPipe |
| `control.primary_hand` | `auto` | `auto` (kanan diutamakan), `right`, `left` |
| `control.target_monitor` | `primary` | `primary` atau `virtual` (semua monitor) |
| `control.click_anchor_ms` | `60` | Kursor dibekukan sesaat saat pinch mulai (lebih kecil = drag lebih cepat) |
| `control.reacquire_ms` | `150` | Durasi luncur halus setelah beku/jeda/scroll |
| `control.sensitivity` | `1.2` | Perbesar area efektif kamera → sudut layar lebih mudah dijangkau |
| `control.active_area.*` | `0.12-0.88` / `0.1-0.8` | Kotak kamera dasar sebelum sensitivitas |
| `control.smoothing.min_cutoff` | `2.5` | Lebih kecil = lebih halus, lebih lag |
| `control.smoothing.beta` | `7.0` | Seberapa cepat filter melepas lag saat gerak cepat |
| `control.smoothing.derivative_cutoff` | `1.8` | Smoothing estimasi kecepatan untuk filter |
| `gestures.pinch_press/release` | `0.36/0.52` | Ambang pinch jempol-telunjuk, dinormalisasi ukuran telapak |
| `gestures.pinch_margin` | `0.75` | Jari pemenang harus ≤0,75× jarak jari pesaing; mencegah klik tertukar |
| `gestures.pinch_stable_ms` | `25` | Pinch harus stabil selama ini sebelum klik dimulai |
| `gestures.click_cooldown_ms` | `150` | Jeda setelah klik sebelum pinch baru boleh mulai |
| `gestures.middle_click_press/release` | `0.18/0.32` | Ambang khusus klik tengah (jempol-jari manis) |
| `gestures.left/right/middle_click` | `true` | Aktif/nonaktifkan jenis klik |
| `gestures.scroll.enter_hold_ms` | `120` | Lama pose V harus ditahan untuk masuk scroll |
| `gestures.scroll.exit_hold_ms` | `150` | Lama pose V hilang sebelum keluar scroll |
| `gestures.scroll.step_px` | `40` | Gerak tangan per satu notch roda mouse |
| `gestures.scroll.smooth` | `0.8` | Smoothing gerak scroll (1 = tanpa smoothing) |
| `gestures.scroll.invert` | `false` | Balik arah scroll |
| `gestures.pause.hold_ms` | `1000` | Lama kepalan ditahan untuk jeda/lanjut |
| `gestures.classifier.enabled` | `false` | Pakai classifier hasil training (bagian 3) |
| `gestures.classifier.model_path` | `models/gesture_classifier.npz` | File model NPZ |
| `gestures.classifier.confidence` | `0.65` | Ambang confidence agar prediksi AI dipakai |
| `gestures.classifier.smoothing` | `0.6` | 0 = mentah; makin tinggi makin halus/lambat |
| `ui.show_preview` | `true` | Tampilkan jendela preview |
| `ui.always_on_top` | `true` | Preview selalu di atas |
| `ui.show_skeleton` | `true` | Gambar skeleton tangan |
| `ui.debug` | `false` | Overlay debug sejak awal |

### Menyesuaikan sensitivitas

Kotak sudut di preview adalah area kamera efektif (setelah `sensitivity`):
area ini yang dipetakan ke seluruh layar.

- Kursor terlalu lari-lari/sensitif: turunkan sensitivitas (`,`) atau perbesar
  kotak (`active_area`), atau turunkan `min_cutoff`.
- Sulit menjangkau sudut layar: naikkan sensitivitas (`.`) ke `1.4-1.7`, atau
  perkecil `active_area` (mis. `0.18-0.82`).
- Klik sulit: naikkan `pinch_press` sedikit (mis. `0.40`) dan `pinch_release`
  (mis. `0.56`). Lihat jarak pinch di overlay `D`.
- Klik tak sengaja: turunkan `pinch_press`, turunkan `pinch_margin` ke `0.7`,
  naikkan `pinch_stable_ms` ke `40`, dan/atau tunjukkan pose netral dulu.
- Klik kiri/kanan tertukar: jauhkan jari yang tidak dipakai saat pinch. HUD
  `JEPIT AMBIGU` berarti jarak kedua jari terlalu dekat. Turunkan
  `pinch_margin` ke `0.7` bila masih tertukar; naikkan ke `0.85` bila klik
  justru sering tidak terjadi.
- Label tangan terbalik: tekan `H`, atau set `camera.swap_handedness`.
- Scroll sulit masuk: tunjukkan V dengan jelas; turunkan
  `scroll.enter_hold_ms` ke `80-100`. Scroll terlalu cepat: naikkan
  `step_px`; terlalu lambat: turunkan.
- Kontrol terasa lag: naikkan `min_cutoff` ke `3.5-4.5`, turunkan
  `process_width` ke `640`, dan pastikan tidak ada aplikasi lain yang
  memakai kamera.

## 5. Troubleshooting

**Kamera hitam / FPS sangat rendah.** Tutup aplikasi lain yang memakai kamera
(Zoom, Teams, OBS, browser). Periksa tombol penutup fisik kamera dan
`Settings > Privacy & security > Camera`. Coba index lain:
`run.bat --camera 1`. Pada mesin uji, kamera terintegrasi sempat hanya
mengirim frame hitam 1 FPS; itu masalah kamera/driver, bukan MediaPipe.
`camera.backend` `msmf` bisa dicoba jika `dshow` bermasalah, dan sebaliknya.

**`Kamera index 0 tidak bisa dibuka`.** Kamera sedang dipakai proses lain,
diblokir kebijakan Windows, atau index salah. `--camera 1`, `2`, ... untuk
mencari kamera lain.

**Kursor bergerak tetapi klik tidak terjadi.** Buka overlay debug (`D`) dan
lihat jarak pinch. Pastikan tangan yang aktif adalah tangan yang Anda pakai
(mode kanan diutamakan). Setelah ganti tangan, tunjukkan pose netral sebentar.
Jika HUD menampilkan `JEPIT AMBIGU`, jauhkan jari yang tidak dipakai. Setelah
klik, tunggu singkat (150 ms) sebelum pinch berikutnya. Beberapa aplikasi
berjalan sebagai administrator; jalankan HandTrack sebagai administrator agar
`SendInput` diterima.

**Klik kiri dan kanan tertukar.** Pinch harus jelas: saat klik kiri, ujung
tengah jangan menempel ke jempol; saat klik kanan, telunjuk jangan ikut
menempel. Aplikasi memakai `pinch_margin` (0,75) sehingga jarak jari pemenang
harus jelas lebih kecil; turunkan ke `0,7` untuk lebih ketat. Cooldown 150 ms
mencegah jari lain langsung memicu klik kedua. Lihat HUD `JEPIT AMBIGU` dan
angka di overlay `D`.

**Scroll susah masuk / `KLIK DIKUNCI`.** Kunci gestur sekarang hanya menahan
**klik** selama ±0,25 dtk—scroll dan kepalan tetap bisa dipakai. Kunci hilang
sendiri setelah 0,25 dtk; tidak perlu mencari pose netral. Tunjukkan V yang
tegas (telunjuk + tengah), salah satu dari jari manis/kelingking terlipat, lalu
tahan ±0,12 dtk. Jika masih sulit, turunkan `gestures.scroll.enter_hold_ms`
ke `80`. Jika scroll malah jalan sendiri, naikkan `enter_hold_ms` atau
mundurkan tangan dari kamera.

**Tangan kiri/kanan tertukar.** Tekan `H` saat preview fokus; HUD akan
menampilkan `LABEL TANGAN DITUKAR`. Untuk permanen, ubah
`camera.swap_handedness` (default di repo ini `true` karena pada mesin ini
label MediaPipe terbalik).

**Sulit menjangkau ujung layar.** Tekan `.` beberapa kali untuk menaikkan
sensitivitas; kotak area di preview akan mengecil. NilaI `1.4-1.7` biasanya
cukup. Turunkan dengan `,` jika kursor malah terlalu sensitif.

**Tidak sengaja jeda.** Jeda sekarang memakai kepalan tangan yang ditahan
±1 detik; telapak terbuka netral. Jangan mengepal terlalu lama saat
mengistirahatkan tangan, atau naikkan `gestures.pause.hold_ms`.

**Klik mengenai jendela preview.** Kursor sedang berada di atas preview yang
selalu di atas. Pindahkan kursor keluar dari area preview, tekan `T` untuk
mematikan always-on-top, atau jalankan dengan `--no-preview`.

**Game tidak merespons.** Banyak game memakai raw input/anti-cheat yang
mengabaikan `SendInput`, terutama mode fullscreen eksklusif. HandTrack
ditujukan untuk aplikasi desktop biasa.

**Model gagal diunduh.** Unduh manual dari
`https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`
lalu simpan sebagai `models/hand_landmarker.task`. Aplikasi memverifikasi
SHA-256 `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`.

**Python 3.13/3.14 tidak didukung MediaPipe.** Gunakan `setup.bat`; script itu
memakai Python 3.12.

**GPU NVIDIA / training.** Build MediaPipe Python yang terpasang
**menonaktifkan GPU** untuk inferensi landmark (`GPU processing is disabled in
build flags`). RTX 5060 dipakai untuk **melatih classifier gestur** lewat
PyTorch CUDA; inferensi classifier sesudahnya hanya NumPy di CPU (ukurannya
kecil). Verifikasi GPU:

```bat
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Jika `False`, pasang build CUDA seperti di bagian 1. Jika `train.bat` bilang
PyTorch tidak ada, ulangi perintah instalasi itu (menjalankan `uv sync` tanpa
group `train` dapat menghapus torch). Model landmark MediaPipe tetap pretrained
dan tidak dilatih ulang; yang dilatih hanyalah MLP kecil di atas 47 fitur
tangan.

## 6. Struktur & pengujian

```
main.py                    entrypoint aplikasi
collect_hand_data.py       entrypoint perekam data
train_gesture.py           entrypoint trainer
collect.bat / train.bat    launcher Windows
handtrack/
  app.py                   loop kamera -> gestur -> mouse -> HUD
  camera.py                buka kamera + resize frame
  classifier.py            fitur 47-D + inferensi NumPy
  cli.py                   argumen command line
  collector.py             perekam landmark + overlay
  config.py                config bertipe + validasi
  cursor.py                pemetaan layar, smoothing, freeze/glide
  filters.py               One Euro filter
  gestures.py              state machine gestur + integrasi classifier
  hands.py                 MediaPipe HandLandmarker + fitur tangan
  input.py                 SendInput Windows + backend dry-run
  mapping.py               konversi koordinat
  overlay.py               HUD OpenCV
  training.py              loader dataset + trainer PyTorch + simpan NPZ
  winapi.py                DPI awareness, batas layar, always-on-top
data/gestures.jsonl        data rekaman (lokal, gitignored)
models/hand_landmarker.task
models/gesture_classifier.npz  hasil training (lokal, gitignored)
config.json
tests/                     unit test tanpa kamera
```

```bat
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff check .
run.bat --dry-run --duration 10
```

Semua logika gestur, filter, mapping, fitur, trainer, dan konfigurasi diuji
dengan landmark sintetis; pengujian kamera sungguhan membutuhkan webcam yang
mengirim gambar. Test trainer otomatis dilewati jika PyTorch tidak terpasang.

## 7. Batasan

- Hanya Windows; macOS/Linux memerlukan backend input lain.
- Hanya mendeteksi tangan yang terlihat jelas (jangan terhalang objek).
- Satu tangan mengendalikan kursor pada satu waktu (dua tangan didukung untuk
  perpindahan otomatis, bukan dua kursor).
- Klik tengah hanya berupa klik (tidak ada hold khusus).
- Model landmark MediaPipe adalah model pretrained dan berjalan di CPU pada
  build ini. Classifier gestur adalah MLP terpisah yang dilatih dari data Anda;
  classifier hanya memilih aksi dan tidak menggeser kursor.
- Kualitas classifier bergantung pada data rekaman. Jika akurasi validasi
  rendah, tambah sesi dan variasi; jangan mencampur dua tangan dalam satu frame.
- Data landmark dan model classifier tetap lokal (`data/`,
  `models/gesture_classifier.npz`) dan diabaikan Git.
- Tidak ada UI kalibrasi; sesuaikan `config.json` atau tombol `H`, `,`, `.`,
  `[`, `]`.
- Kontrol mouse tidak akan bekerja pada aplikasi elevated jika HandTrack tidak
  dijalankan elevated.

## 8. Kredit

Deteksi tangan memakai [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
(Google, Apache License 2.0). Model `hand_landmarker.task` disertakan dari
distribusi resmi MediaPipe; detail dan hash ada di `models/README.md`.
