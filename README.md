# HandTrack

Kontrol mouse Windows sepenuhnya dengan gestur tangan dari webcam. Cukup satu
tangan; webcam tetap melacak dua tangan agar Anda bisa berpindah tangan tanpa
memutus kontrol.

- **Kursor** mengikuti buku jari telunjuk, dipetakan ke layar dengan filter
  One Euro yang dibuat lebih responsif (halus saat diam, cepat saat bergerak).
- **Klik kiri** dengan pinch jempol + telunjuk sampai ujung jari **menempel**;
  satu klik bersih dikirim saat kontak, tanpa tombol tertahan. **Hold/drag**
  dengan pinch **hampir menempel** (ada celah kecil) dan tahan ±300 ms, lalu
  geser tangan.
- **Klik kanan** dengan pinch jempol + jari tengah.
- **Klik tengah** (opsional) dengan pinch rapat jempol + jari manis.
- **Scroll** dengan pose dua jari (telunjuk + tengah) lalu gerakkan tangan
  naik/turun.
- **Jeda / lanjut** dengan mengepalkan tangan dan menahannya sebentar, atau
  tombol `P`.
- **Klik ganda**: dua pinch-tempel cepat. Kursor ditahan di titik klik sebentar
  supaya kedua klik mendarat di posisi yang sama.
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
| Pinch jempol + telunjuk sampai ujung menempel | Klik kiri (satu klik bersih saat kontak) |
| Pinch jempol + telunjuk hampir menempel, tahan ≥300 ms | Hold/drag; geser tangan lalu lepas |
| Dua pinch-tempel cepat | Double-click |
| Pinch jempol + jari tengah | Klik kanan (lepas pinch untuk mengeksekusi) |
| Pinch rapat jempol + jari manis | Klik tengah (opsional) |
| Pose V (telunjuk + tengah), tahan ±0,12 dtk | Masuk mode scroll; gerakkan tangan naik/turun |
| Kepal tangan, tahan ±1 dtk | Jeda; ulangi untuk lanjut |
| Telapak terbuka | Netral / tidak melakukan aksi |
| Kepalan (sebentar) | Tidak mengklik (mencegah klik tak sengaja) |

Catatan:

- Jeda tidak lagi memakai telapak terbuka—telapak sekarang netral. Pakai
  kepalan yang ditahan sekitar 1 detik.
- **Klik kiri vs hold.** Ada dua zona pinch:
  - **Menempel** (`touch_press`, default 0,18): aplikasi langsung mengirim
    down+up **satu kali** di titik itu. Jari yang tetap menempel tidak
    mengulang klik dan tidak berubah menjadi hold.
  - **Hampir menempel** (`touch_release`..`pinch_press`, default 0,26..0,36):
    tahan ±300 ms (`gestures.left_hold_ms`) untuk menekan tombol kiri dan
    drag. Kursor dibekukan selama fase menunggu; HUD menampilkan progress
    `NEMPEL = KLIK | TAHAN HAMPIR = DRAG`.
  - Untuk klik kedua, buka jari melewati `touch_release` lalu tempelkan lagi.
    Klik kedua boleh terjadi dalam pinch yang sama (double pinch).
- Saat pinch jempol + jari tengah, jaga agar ujung telunjuk tidak ikut
  menyentuh jempol. Aplikasi memilih jari yang **jelas paling dekat**; kalau
  jarak telunjuk dan tengah hampir sama, klik sengaja tidak dijalankan dan HUD
  menampilkan `JEPIT AMBIGU`. Setelah pinch baru, ada jeda 100 ms sebelum
  pinch berikutnya boleh mulai; pinch yang sudah menempel setelah jeda boleh
  lanjut.
- Klik menempel tetap menahan kursor ±60 ms setelah klik
  (`control.click_anchor_ms`) supaya double-click mendarat di titik yang sama.
- Setelah satu sentuhan klik, hold sengaja diblokir sampai pinch dilepas.
  Untuk mulai hold/drag, buka tangan dulu lalu lakukan pinch
  **hampir-menempel** yang baru. Pinch dekat yang dilepas tanpa pernah
  menempel tidak menghasilkan klik. Kursor juga tetap di anchor selama pinch
  masih aktif; buka pinch melewati `pinch_release` untuk menggerakkannya lagi.
- Klik memakai aturan ketat: jari pemenang harus unik (margin 0,75), posisi
  pinch harus stabil ±25 ms, dan sentuhan harus melewati `touch_press`.
  Pinch yang dimulai oleh AI dilepas berdasarkan jarak jari (bukan menunggu
  label AI yang halus).
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
| `2` | label `left` (pinch jempol + telunjuk: tempel **dan** hampir) |
| `3` | label `right` (pinch jempol + tengah) |
| `4` | label `middle` (pinch rapat jempol + jari manis) |
| `5` | label `fist` (kepalan untuk jeda) |
| `6` | label `scroll` (pose V: telunjuk + tengah terbuka) |
| `Space` | mulai / berhenti merekam |
| `N` | sesi baru |
| `Q` / `Esc` | keluar |

Ganti label otomatis menjeda rekaman supaya data tidak tercampur. Collector
default memakai target **600 frame per label per sesi** dan berhenti otomatis
saat target tercapai. Untuk `left`, target dipecah dua blok: **300 menempel +
300 hampir-menempel**. Rekam dengan variasi jarak tangan, posisi di frame,
sudut, dan pencahayaan.

Lakukan **minimal 3 sesi lengkap** (`N`), dan isi **semua 6 label di setiap
sesi**. HUD menampilkan `target` dan `sesi x/6 label`; jangan menekan `N`
sebelum 6/6. Trainer memakai sesi sebagai pembatas validasi; sesi yang tidak
lengkap membuat akurasi tidak jujur (trainer akan memperingatkan dan memakai
split frame yang lebih optimistis).

Untuk `left`, rekam **dua variasi dalam label yang sama**:

- ujung jempol + telunjuk **menempel** (klik) — blok pertama, auto-stop 300,
- **hampir menempel** dengan celah kecil (hold/drag) — blok kedua, auto-stop 600.

Pose yang direkam:

- `neutral`: tangan santai/terbuka atau menunjuk seperti saat menggerakkan
  kursor; bukan pinch, bukan kepalan, bukan pose V.
- `left`: pinch jempol + telunjuk, campur pose tempel (klik) dan hampir
  (hold) sekitar 50:50. Jangan hanya salah satu.
- `right` / `middle`: pinch yang Anda maksud untuk tiap klik.
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

Trainer memakai GPU jika tersedia, menyeimbangkan kelas, dan menyimpan
`models/gesture_classifier.npz`. Sebelum training, ia memeriksa data yang
relevan untuk gestur sekarang:

- jumlah sesi yang berisi semua label,
- proporsi frame `left` yang **menempel** vs **hampir-menempel**,
- kelas yang belum ada atau sampelnya sedikit.

Secara default trainer sekarang memakai **ensemble 5 MLP + label smoothing
0,05**. Kelima model dilatih dengan seed berbeda; probabilitasnya dirata-rata
saat inferensi sehingga keputusan lebih stabil. Model tetap MLP/features yang
sama, hanya bobotnya disimpan sebagai tumpukan 5 model.

Validasi memakai **sesi terpisah** bila memungkinkan:

- jika tiap sesi berisi semua label, satu/lebih sesi dipakai sebagai validasi;
- jika tiap sesi hanya satu label tetapi tiap kelas punya >=2 sesi, trainer
  memakai satu sesi per kelas sebagai validasi (tetap sesi-terpisah);
- jika tidak memungkinkan, ia memakai potongan **kronologis** tiap kelas
  (bukan frame acak) dan memberi peringatan bahwa angkanya optimistis.

Contoh keluaran:

```
Ensemble     : 5 model | label smoothing 0.05
Val per model: 95.1%, 95.3%, 95.0%, 95.2%, 95.4%
Akurasi val  : 95.7%
Validasi     : sesi terpisah
Data left    : tempel 41% | hampir 32%
Metrik per kelas (support / recall / precision / F1):
  left     n= 470  R= 91.5%  P= 89.2%  F1= 90.3%
  ...
```

Perhatikan `R`/`P`/`F1` per kelas. Jika satu kelas lemah, confusion matrix
menunjukkan label apa yang tertukar; tambah variasi pose untuk label itu.
Jangan hanya melihat akurasi total. Validasi frame dari satu rekaman bisa
mendekati 100% meskipun generalisasi ke rekaman lain lebih rendah; karena itu
rekam beberapa sesi/take yang benar-benar berbeda.

Opsi lain:

```bat
train.bat --data data/gestures.jsonl --out models/gesture_classifier.npz --epochs 200
train.bat --ensemble 5 --label-smoothing 0.05
train.bat --config config.json --epochs 200
```

`--ensemble 1` memakai satu model; `--label-smoothing 0` mematikannya.

`--config` dipakai untuk ambang `touch_press`/`touch_release` pada pemeriksaan
kualitas dan tidak mengubah arsitektur model.

### 3.3 Aktifkan classifier

Ubah `config.json`:

```json
"gestures": {
  "classifier": {
    "enabled": true,
    "model_path": "models/gesture_classifier.npz",
    "confidence": 0.65,
    "smoothing": 0.7
  }
}
```

Jalankan ulang `run.bat`. Banner akan menampilkan `AI gestur: aktif`; overlay
debug (`D`) menampilkan `ai=label confidence`. Jika model tidak ada, aplikasi
memberi peringatan dan tetap memakai aturan geometri.

Cara kerja saat aktif:

- Prediksi dengan confidence ≥ `confidence` menggantikan pilihan pinch/fist/
  scroll. Label `neutral` berarti kembali ke aturan geometri.
- Pinch yang dimulai oleh AI dilepas saat label AI berubah **atau** jarak jari
  membuka melewati ambang (dimulai dari jarak saat pinch). Ini mencegah klik
  singkat tertahan karena smoothing label AI. Kehilangan tangan tetap melepas
  semua tombol.
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
| `control.click_anchor_ms` | `60` | Kursor ditahan setelah tap/hold diputuskan; lebih kecil = gerak lebih cepat |
| `control.reacquire_ms` | `150` | Durasi luncur halus setelah beku/jeda/scroll |
| `control.sensitivity` | `1.2` | Perbesar area efektif kamera → sudut layar lebih mudah dijangkau |
| `control.active_area.*` | `0.12-0.88` / `0.1-0.8` | Kotak kamera dasar sebelum sensitivitas |
| `control.smoothing.min_cutoff` | `2.5` | Lebih kecil = lebih halus, lebih lag |
| `control.smoothing.beta` | `7.0` | Seberapa cepat filter melepas lag saat gerak cepat |
| `control.smoothing.derivative_cutoff` | `1.8` | Smoothing estimasi kecepatan untuk filter |
| `gestures.pinch_press/release` | `0.36/0.52` | Ambang pinch jempol-telunjuk, dinormalisasi ukuran telapak |
| `gestures.pinch_margin` | `0.75` | Jari pemenang harus ≤0,75× jarak jari pesaing; mencegah klik tertukar |
| `gestures.touch_press` | `0.18` | Jari dianggap **menempel** (klik) bila jarak pinch ≤ nilai ini |
| `gestures.touch_release` | `0.26` | Jari dianggap lepas dari sentuhan di atas nilai ini (histeresis) |
| `gestures.pinch_stable_ms` | `25` | Pinch harus stabil selama ini sebelum klik dimulai |
| `gestures.left_hold_ms` | `300` | Lama menahan zona hampir-menempel sebelum masuk hold/drag |
| `gestures.click_cooldown_ms` | `100` | Jeda singkat setelah klik sebelum pinch baru boleh mulai |
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
| `gestures.classifier.smoothing` | `0.7` | 0 = mentah; makin tinggi makin halus/lambat |
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
- Jari sudah menempel tetapi tidak klik: naikkan `gestures.touch_press`
  sedikit (mis. `0.20-0.22`); jika klik malah terpicu saat hampir menempel,
  turunkan. `touch_release` harus di antara `touch_press` dan `pinch_press`.
- Hold sulit/tidak sengaja: turunkan atau naikkan `gestures.left_hold_ms`
  (mis. `250` lebih cepat, `350` lebih sabar). Jaga celah kecil di zona
  hampir-menempel; jangan sampai ujung jari menempel.
- Klik kedua/double-click sulit: pastikan jari membuka melewati
  `touch_release` sebelum menempel lagi, dan turunkan `click_cooldown_ms`
  ke `70-80`.
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
lihat jarak pinch. Klik kiri hanya dikirim saat ujung jempol dan telunjuk
benar-benar menempel (jarak ≤ `gestures.touch_press`, default 0,18). Jika
jarak Anda masih di atas itu, tempelkan sedikit lagi atau naikkan
`touch_press` ke 0,20-0,22. Setelah klik, buka jari melewati `touch_release`
(0,26) sebelum menempel lagi. `JEPIT AMBIGU` berarti jari lain ikut dekat;
jauhkan. Beberapa aplikasi berjalan sebagai administrator; jalankan HandTrack
sebagai administrator agar `SendInput` diterima.

**Klik kiri dan kanan tertukar.** Pinch harus jelas: saat klik kiri, ujung
tengah jangan menempel ke jempol; saat klik kanan, telunjuk jangan ikut
menempel. Aplikasi memakai `pinch_margin` (0,75) sehingga jarak jari pemenang
harus jelas lebih kecil; turunkan ke `0,7` untuk lebih ketat. Cooldown 100 ms
mencegah jari lain langsung memicu klik kedua. Lihat HUD `JEPIT AMBIGU` dan
angka di overlay `D`.

**Klik biasa malah jadi hold/drag.** Berarti jari berhenti di zona
**hampir-menempel** (tidak menempel) selama ≥300 ms. Untuk klik, tempelkan
ujung jari sampai menyentuh; klik dikirim saat kontak, bukan saat ditahan.
Untuk hold, jaga celah kecil dan tahan. Jika transisi menuju kontak terlalu
lambat, turunkan `gestures.left_hold_ms` ke `250-280` atau besarkan
`touch_press`.

**Double-click sulit.** Lakukan dua **sentuhan** cepat: tempelkan jempol +
telunjuk, buka jari melewati `touch_release`, lalu tempelkan lagi. Setiap
sentuhan mengirim down+up penuh, dan kursor ditahan `control.click_anchor_ms`
(60 ms) agar kedua klik berada di titik yang sama. Setelah sentuhan, hold
diblokir sampai pinch dilepas; jadi jangan berhenti di zona hampir-menempel
saat ingin double-click.

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
