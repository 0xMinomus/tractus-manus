"""Dataset JSONL + trainer MLP PyTorch untuk classifier gestur.

Data direkam oleh collect_hand_data.py. Model disimpan sebagai NPZ yang
di-infer murni dengan NumPy oleh handtrack/classifier.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .classifier import CLASSES, FEATURE_DIM, extract_features_from_landmarks
from .config import load_config
from .hands import PROJECT_ROOT, Hand, Pinch, analyze_hand

log = logging.getLogger(__name__)

DEFAULT_DATA = PROJECT_ROOT / "data" / "gestures.jsonl"
DEFAULT_CONFIG = PROJECT_ROOT / "config.json"
DEFAULT_OUT = PROJECT_ROOT / "models" / "gesture_classifier.npz"
HIDDEN1 = 128
HIDDEN2 = 64


@dataclass(frozen=True)
class Dataset:
    features: np.ndarray
    labels: np.ndarray
    sessions: np.ndarray
    classes: tuple[str, ...]
    counts: dict[str, int]
    index_pinch: np.ndarray | None = None


@dataclass(frozen=True)
class QualityReport:
    warnings: tuple[str, ...]
    session_count: int
    complete_sessions: int
    left_touch_fraction: float | None
    left_near_fraction: float | None


@dataclass(frozen=True)
class TrainingReport:
    out_path: Path
    device: str
    epochs_run: int
    best_val_loss: float
    train_accuracy: float
    val_accuracy: float
    confusion: np.ndarray
    train_sessions: tuple[str, ...]
    val_sessions: tuple[str, ...]
    classes: tuple[str, ...]
    split_mode: str = "session"
    ensemble_size: int = 1
    label_smoothing: float = 0.0
    member_val_accuracy: tuple[float, ...] = ()
    member_epochs: tuple[int, ...] = ()


@dataclass(frozen=True)
class MemberResult:
    state: dict[str, object]
    val_loss: float
    epochs_run: int
    train_accuracy: float
    val_accuracy: float
    val_probs: np.ndarray


def load_dataset(path: Path | str) -> Dataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data tidak ditemukan: {path}. Jalankan collect_hand_data.py dulu."
        )
    features: list[np.ndarray] = []
    labels: list[str] = []
    sessions: list[str] = []
    counts: Counter[str] = Counter()
    index_pinch: list[float] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                label = str(rec["label"])
                session = str(rec["session"])
                hand = str(rec["hand"])
                landmarks = np.asarray(rec["landmarks"], dtype=np.float64)
                width = int(rec.get("width", 1280))
                height = int(rec.get("height", 720))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_no}: record tidak valid ({exc})") from exc
            if hand not in ("Left", "Right"):
                raise ValueError(f"{path}:{line_no}: hand harus Left/Right")
            if landmarks.shape != (21, 3):
                raise ValueError(
                    f"{path}:{line_no}: landmarks harus (21,3), dapat {landmarks.shape}"
                )
            if not np.isfinite(landmarks[:, :2]).all():
                raise ValueError(f"{path}:{line_no}: landmark tidak finite")
            if width <= 0 or height <= 0:
                width, height = 1280, 720
            features.append(extract_features_from_landmarks(landmarks, hand))
            pinch = analyze_hand(Hand(hand, 1.0, landmarks, width, height)).pinch
            index_pinch.append(float(pinch.get(Pinch.INDEX, float("nan"))))
            labels.append(label)
            sessions.append(session)
            counts[label] += 1

    if not labels:
        raise ValueError(f"Tidak ada sampel di {path}")
    seen = set(labels)
    classes = [c for c in CLASSES if c in seen]
    classes.extend(sorted(seen - set(classes)))
    index = {name: i for i, name in enumerate(classes)}
    unknown = [c for c in counts if c not in CLASSES]
    if unknown:
        log.warning("Label di luar daftar standar: %s", unknown)
    log.info("Data: %d sampel, %d sesi", len(labels), len(set(sessions)))
    log.info("Per label: %s", dict(sorted(counts.items())))
    return Dataset(
        features=np.stack(features),
        labels=np.array([index[label] for label in labels], dtype=np.int64),
        sessions=np.array(sessions),
        classes=tuple(classes),
        counts=dict(counts),
        index_pinch=np.array(index_pinch, dtype=np.float32),
    )


def check_dataset(
    dataset: Dataset,
    *,
    touch_press: float,
    touch_release: float,
    pinch_press: float,
) -> QualityReport:
    """Periksa data yang sekarang relevan untuk klik-tempel vs hold-dekat."""
    classes = list(dataset.classes)
    labels = np.asarray(dataset.labels)
    sessions = np.asarray(dataset.sessions)
    warnings: list[str] = []
    unique_sessions = np.unique(sessions)
    all_indices = set(range(len(classes)))
    complete_sessions = 0
    for session in unique_sessions:
        present = set(labels[sessions == session].tolist())
        if present == all_indices:
            complete_sessions += 1

    can_session = len(unique_sessions) >= 2 and all(
        len(set(sessions[labels == cls].tolist())) >= 2
        for cls in range(len(classes))
    )
    if len(unique_sessions) == 1:
        warnings.append(
            "Hanya ada 1 sesi. Validasi memakai split frame per kelas; akurasi "
            "tidak mengukur generalisasi. Rekam beberapa sesi."
        )
    elif complete_sessions < len(unique_sessions):
        if can_session:
            warnings.append(
                "Tidak ada sesi yang memuat semua label, tetapi setiap kelas "
                "punya >=2 sesi; validasi memakai sesi-per-kelas (tetap "
                "terpisah). Idealnya tetap rekam semua label dalam satu sesi."
            )
        else:
            warnings.append(
                f"{len(unique_sessions) - complete_sessions}/{len(unique_sessions)} "
                "sesi tidak berisi semua label dan ada kelas dengan <2 sesi. "
                "Trainer memakai split frame kronologis (akurasi optimistis)."
            )

    missing = [name for name in CLASSES if name not in classes]
    if missing:
        warnings.append(
            f"Label belum ada: {', '.join(missing)}. Model tidak bisa "
            "memprediksi label tersebut; rekam semua label bila dipakai."
        )
    small = [name for name in classes if dataset.counts.get(name, 0) < 30]
    if small:
        warnings.append(
            f"Sampel <30 untuk: {', '.join(small)}. Tambah durasi rekaman."
        )

    touch_fraction: float | None = None
    near_fraction: float | None = None
    if "left" in classes and dataset.index_pinch is not None:
        left_index = classes.index("left")
        distances = np.asarray(dataset.index_pinch)[labels == left_index]
        distances = distances[np.isfinite(distances)]
        if distances.size:
            touch_fraction = float(np.mean(distances <= touch_press))
            near_fraction = float(
                np.mean((distances > touch_release) & (distances <= pinch_press))
            )
            if touch_fraction < 0.15:
                warnings.append(
                    f"Data left: hanya {touch_fraction * 100:.0f}% frame dengan "
                    "ujung jari menempel. Klik butuh contoh pose tempel; rekam "
                    "juga variasi klik (ujung jari benar-benar bertemu)."
                )
            if near_fraction < 0.15:
                warnings.append(
                    f"Data left: hanya {near_fraction * 100:.0f}% frame di zona "
                    "hampir-menempel. Hold butuh contoh pose celah kecil; rekam "
                    "juga variasi hold tanpa menempel."
                )

    return QualityReport(
        warnings=tuple(warnings),
        session_count=len(unique_sessions),
        complete_sessions=complete_sessions,
        left_touch_fraction=touch_fraction,
        left_near_fraction=near_fraction,
    )


def split_train_val(
    labels: np.ndarray,
    sessions: np.ndarray,
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Split train/val; sesi-disjoint bila memungkinkan, jika tidak stratified."""
    labels = np.asarray(labels)
    sessions = np.asarray(sessions)
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction harus di antara 0 dan 1")
    classes = np.unique(labels)
    unique_sessions = np.unique(sessions)
    can_session = len(unique_sessions) >= 2
    if can_session:
        for cls in classes:
            if len(set(sessions[labels == cls].tolist())) < 2:
                can_session = False
                break
    if can_session:
        rng = np.random.default_rng(seed)
        n_val = max(1, int(round(len(unique_sessions) * val_fraction)))
        n_val = min(n_val, len(unique_sessions) - 1)
        for _ in range(200):
            shuffled = unique_sessions.copy()
            rng.shuffle(shuffled)
            val_sessions = set(shuffled[:n_val].tolist())
            val_mask = np.isin(sessions, list(val_sessions))
            train_mask = ~val_mask
            if not val_mask.any() or not train_mask.any():
                continue
            if (
                set(labels[val_mask].tolist()) == set(classes.tolist())
                and set(labels[train_mask].tolist()) == set(classes.tolist())
            ):
                return train_mask, val_mask, "session"
        # Cadangan: pilih satu sesi per kelas sebagai validasi. Ini penting
        # untuk data yang tiap sesinya hanya satu label tetapi tiap kelas
        # memiliki beberapa sesi (beberapa kali rekam).
        chosen_sessions: list[str] = []
        for cls in classes:
            options = sorted(set(sessions[labels == cls].tolist()))
            rng.shuffle(options)
            chosen_sessions.append(options[0])
        val_sessions = set(chosen_sessions)
        val_mask = np.isin(sessions, list(val_sessions))
        train_mask = ~val_mask
        if (
            val_mask.any()
            and train_mask.any()
            and set(labels[val_mask].tolist()) == set(classes.tolist())
            and set(labels[train_mask].tolist()) == set(classes.tolist())
        ):
            return train_mask, val_mask, "session"
        log.warning(
            "Tidak ada pembagian sesi yang memuat semua kelas di train dan "
            "validasi; memakai stratified frame split."
        )

    # Fallback: ambil ekor kronologis tiap kelas, bukan frame acak. Frame
    # acak dari satu rekaman kontinu hampir identik dengan tetangganya dan
    # membuat akurasi validasi terlalu optimistis.
    val_mask = np.zeros(len(labels), dtype=bool)
    for cls in classes:
        indices = np.flatnonzero(labels == cls)
        if len(indices) < 2:
            continue
        n_val = max(1, int(round(len(indices) * val_fraction)))
        n_val = min(n_val, len(indices) - 1)
        val_mask[indices[-n_val:]] = True
    if not val_mask.any():
        counts = np.bincount(labels)
        cls = int(np.argmax(counts))
        indices = np.flatnonzero(labels == cls)
        if len(indices) >= 2:
            val_mask[indices[0]] = True
    train_mask = ~val_mask
    if not train_mask.any():
        raise ValueError("Split data gagal: data latih kosong")
    return train_mask, val_mask, "stratified"


def _mlp(torch, n_in: int, n_out: int):
    class MLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(n_in, HIDDEN1)
            self.fc2 = torch.nn.Linear(HIDDEN1, HIDDEN2)
            self.fc3 = torch.nn.Linear(HIDDEN2, n_out)
            self.drop = torch.nn.Dropout(0.1)

        def forward(self, x):
            x = torch.nn.functional.relu(self.fc1(x))
            x = self.drop(x)
            x = torch.nn.functional.relu(self.fc2(x))
            return self.fc3(x)

    return MLP()


def _fit_member(
    torch,
    *,
    xtr,
    ytr,
    xval,
    yval,
    weight_t,
    n_classes: int,
    device: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    label_smoothing: float,
    tag: str,
) -> MemberResult:
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = _mlp(torch, FEATURE_DIM, n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=weight_t, label_smoothing=label_smoothing
    )
    best_loss = float("inf")
    best_state = None
    bad = 0
    epochs_run = 0
    n_train = int(xtr.shape[0])

    for epoch in range(1, epochs + 1):
        epochs_run = epoch
        model.train()
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xtr[idx])
            loss = loss_fn(logits, ytr[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(xval)
            val_loss = float(loss_fn(val_logits, yval).item())
            val_acc = float((val_logits.argmax(dim=1) == yval).float().mean().item())
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        if epoch == 1 or epoch % 10 == 0 or bad == 0:
            log.info(
                "%s epoch %3d  val_loss %.4f  val_acc %.3f%s",
                tag,
                epoch,
                val_loss,
                val_acc,
                "  *" if bad == 0 else "",
            )
        if bad >= patience:
            log.info("%s early stop di epoch %d", tag, epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_pred = model(xtr).argmax(dim=1)
        val_logits = model(xval)
        val_pred = val_logits.argmax(dim=1)
        val_probs = torch.softmax(val_logits, dim=1)
        train_acc = float((train_pred == ytr).float().mean().item())
        val_acc = float((val_pred == yval).float().mean().item())
        val_loss = float(loss_fn(val_logits, yval).item())

    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    return MemberResult(
        state=state,
        val_loss=val_loss,
        epochs_run=epochs_run,
        train_accuracy=train_acc,
        val_accuracy=val_acc,
        val_probs=val_probs.cpu().numpy(),
    )


def train_model(
    dataset: Dataset,
    out_path: Path | str = DEFAULT_OUT,
    *,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
    val_fraction: float = 0.2,
    patience: int = 30,
    ensemble: int = 1,
    label_smoothing: float = 0.0,
) -> TrainingReport:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - lingkungan tanpa torch
        raise RuntimeError(
            "PyTorch belum terpasang. Pasang dulu, misalnya:\n"
            "  uv pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
            "atau versi CPU:  uv sync --group train"
        ) from exc

    x = np.asarray(dataset.features, dtype=np.float32)
    y = np.asarray(dataset.labels, dtype=np.int64)
    n_classes = len(dataset.classes)
    if x.ndim != 2 or x.shape[1] != FEATURE_DIM:
        raise ValueError(f"Fitur harus (n,{FEATURE_DIM}), dapat {x.shape}")
    if y.min() < 0 or y.max() >= n_classes:
        raise ValueError("Label di luar kelas dataset")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs dan batch harus >= 1")
    if not (1 <= ensemble <= 8):
        raise ValueError("ensemble harus 1..8")
    if not (0.0 <= label_smoothing <= 0.2):
        raise ValueError("label_smoothing harus 0..0.2")

    train_mask, val_mask, split_mode = split_train_val(
        y, dataset.sessions, val_fraction=val_fraction, seed=seed
    )
    val_sessions = set(dataset.sessions[val_mask].tolist())
    train_sessions = set(dataset.sessions[train_mask].tolist())
    if split_mode == "session":
        log.info(
            "Validasi sesi terpisah: train=%s, val=%s",
            sorted(train_sessions),
            sorted(val_sessions),
        )
    else:
        log.warning(
            "Validasi memakai stratified frame split (bukan sesi terpisah); "
            "akurasi bisa lebih optimistis. Rekam semua label dalam tiap sesi."
        )

    train_classes = set(int(v) for v in np.unique(y[train_mask]))
    val_classes = set(int(v) for v in np.unique(y[val_mask]))
    missing = [dataset.classes[i] for i in train_classes if i not in val_classes]
    if missing:
        log.warning("Kelas yang tidak ada di validasi: %s", missing)
    unavailable = [dataset.classes[i] for i in val_classes if i not in train_classes]
    if unavailable:
        log.warning("Kelas validasi tanpa contoh latih (tidak mungkin diprediksi): %s", unavailable)

    mean = x[train_mask].mean(axis=0)
    std = x[train_mask].std(axis=0)
    std[std < 1e-6] = 1.0
    x = (x - mean) / std

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))
    else:
        log.info("GPU tidak tersedia; training memakai CPU.")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    xt = torch.from_numpy(x).to(device)
    yt = torch.from_numpy(y).to(device)
    xtr, ytr = xt[train_mask], yt[train_mask]
    xval, yval = xt[val_mask], yt[val_mask]

    counts = np.bincount(y[train_mask], minlength=n_classes).astype(np.float64)
    present = counts > 0
    weights = np.zeros(n_classes, dtype=np.float32)
    weights[present] = counts.sum() / (present.sum() * counts[present])
    weight_t = torch.tensor(weights, dtype=torch.float32, device=device)

    log.info(
        "Ensemble: %d model, label smoothing %.2f",
        ensemble,
        label_smoothing,
    )
    member_results: list[MemberResult] = []
    for member in range(ensemble):
        member_seed = seed + member
        log.info("Melatih model %d/%d (seed %d)", member + 1, ensemble, member_seed)
        member_results.append(
            _fit_member(
                torch,
                xtr=xtr,
                ytr=ytr,
                xval=xval,
                yval=yval,
                weight_t=weight_t,
                n_classes=n_classes,
                device=device,
                seed=member_seed,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                patience=patience,
                label_smoothing=label_smoothing,
                tag=f"m{member + 1}",
            )
        )

    val_probs = np.mean([result.val_probs for result in member_results], axis=0)
    y_true = yval.detach().cpu().numpy()
    val_pred = val_probs.argmax(axis=1)
    val_acc = float(np.mean(val_pred == y_true))
    train_acc = float(np.mean([result.train_accuracy for result in member_results]))
    best_val_loss = float(np.mean([result.val_loss for result in member_results]))
    epochs_run = max(result.epochs_run for result in member_results)

    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true, pred in zip(y_true, val_pred, strict=True):
        confusion[int(true), int(pred)] += 1

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def weight_stack(key: str) -> np.ndarray:
        return np.stack(
            [result.state[key].numpy().T for result in member_results], axis=0
        )

    def bias_stack(key: str) -> np.ndarray:
        return np.stack(
            [result.state[key].numpy() for result in member_results], axis=0
        )

    np.savez(
        out_path,
        W1=weight_stack("fc1.weight"),
        b1=bias_stack("fc1.bias"),
        W2=weight_stack("fc2.weight"),
        b2=bias_stack("fc2.bias"),
        W3=weight_stack("fc3.weight"),
        b3=bias_stack("fc3.bias"),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        classes=np.array(dataset.classes, dtype=str),
        ensemble=np.int64(ensemble),
        label_smoothing=np.float64(label_smoothing),
        format_version=np.int64(2),
    )
    return TrainingReport(
        out_path=out_path,
        device=device,
        epochs_run=epochs_run,
        best_val_loss=best_val_loss,
        train_accuracy=train_acc,
        val_accuracy=val_acc,
        confusion=confusion,
        train_sessions=tuple(sorted(set(dataset.sessions[train_mask]))),
        val_sessions=tuple(sorted(val_sessions)),
        classes=dataset.classes,
        split_mode=split_mode,
        ensemble_size=ensemble,
        label_smoothing=label_smoothing,
        member_val_accuracy=tuple(result.val_accuracy for result in member_results),
        member_epochs=tuple(result.epochs_run for result in member_results),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_gesture",
        description="Latih classifier gestur dari data landmark JSONL.",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--ensemble",
        type=int,
        default=5,
        help="jumlah MLP yang dilatih dan dirata-ratakan (1-8; default 5)",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="label smoothing 0..0.2 (default 0.05)",
    )
    return parser


def class_metrics(
    confusion: np.ndarray, classes: tuple[str, ...]
) -> list[tuple[str, int, float, float, float]]:
    """Support, recall, precision, F1 per kelas dari confusion matrix."""
    rows: list[tuple[str, int, float, float, float]] = []
    total = int(confusion.sum())
    for index, name in enumerate(classes):
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        tp = int(confusion[index, index])
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        rows.append((name, support, recall, precision, f1))
    if total == 0:  # pragma: no cover - dipakai untuk pesan
        log.warning("Confusion matrix kosong")
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        cfg = load_config(args.config)
        dataset = load_dataset(args.data)
        quality = check_dataset(
            dataset,
            touch_press=cfg.gestures.touch_press,
            touch_release=cfg.gestures.touch_release,
            pinch_press=cfg.gestures.pinch_press,
        )
        for warning in quality.warnings:
            log.warning("%s", warning)
        log.info(
            "Sesi lengkap (semua label): %d/%d",
            quality.complete_sessions,
            quality.session_count,
        )
        if quality.left_touch_fraction is not None:
            log.info(
                "Data left: tempel %.0f%%, hampir-menempel %.0f%%",
                quality.left_touch_fraction * 100,
                (quality.left_near_fraction or 0.0) * 100,
            )
        report = train_model(
            dataset,
            args.out,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            seed=args.seed,
            val_fraction=args.val_fraction,
            ensemble=args.ensemble,
            label_smoothing=args.label_smoothing,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        return 0

    print()
    print("=" * 62)
    print("  Training selesai")
    print("=" * 62)
    print(f"  Model        : {report.out_path}")
    print(f"  Device       : {report.device}")
    print(f"  Epoch        : {report.epochs_run}")
    print(
        f"  Ensemble     : {report.ensemble_size} model "
        f"| label smoothing {report.label_smoothing:.2f}"
    )
    if report.member_val_accuracy:
        members = ", ".join(
            f"{value * 100:.1f}%" for value in report.member_val_accuracy
        )
        print(f"  Val per model: {members}")
    print(f"  Akurasi train: {report.train_accuracy * 100:.1f}%")
    print(f"  Akurasi val  : {report.val_accuracy * 100:.1f}%")
    split_text = (
        "sesi terpisah"
        if report.split_mode == "session"
        else "stratified frame (optimistis)"
    )
    print(f"  Validasi     : {split_text}")
    print(f"  Sesi lengkap : {quality.complete_sessions}/{quality.session_count}")
    if quality.left_touch_fraction is not None:
        print(
            "  Data left    : "
            f"tempel {quality.left_touch_fraction * 100:.0f}% | "
            f"hampir {((quality.left_near_fraction or 0.0) * 100):.0f}%"
        )
    print(f"  Sesi train   : {', '.join(report.train_sessions)}")
    print(f"  Sesi val     : {', '.join(report.val_sessions)}")
    print("  Confusion (baris=asli, kolom=prediksi):")
    for i, name in enumerate(report.classes):
        row = " ".join(f"{v:5d}" for v in report.confusion[i])
        print(f"    {name:8s} {row}")
    print("  Metrik per kelas (support / recall / precision / F1):")
    weak: list[str] = []
    for name, support, recall, precision, f1 in class_metrics(
        report.confusion, report.classes
    ):
        if support == 0:
            print(f"    {name:8s} support=0 (tidak ada di validasi)")
            continue
        print(
            f"    {name:8s} n={support:4d}  "
            f"R={recall * 100:5.1f}%  P={precision * 100:5.1f}%  "
            f"F1={f1 * 100:5.1f}%"
        )
        if recall < 0.7 or precision < 0.7:
            weak.append(name)
    if weak:
        print(
            "  Perhatian    : kelas lemah "
            f"{', '.join(weak)}. Tambah data variasi pose; cek confusion di atas."
        )
    print()
    print("  Aktifkan di config.json:")
    print('    "gestures": { "classifier": { "enabled": true } }')
    print("  Lalu jalankan ulang: run.bat")
    print("=" * 62)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
