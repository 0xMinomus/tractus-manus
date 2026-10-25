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
from .hands import PROJECT_ROOT

log = logging.getLogger(__name__)

DEFAULT_DATA = PROJECT_ROOT / "data" / "gestures.jsonl"
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
            features.append(extract_features_from_landmarks(landmarks, hand))
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
    )


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

    rng = np.random.default_rng(seed)
    sessions = np.unique(dataset.sessions)
    if len(sessions) >= 2:
        shuffled = list(sessions)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_fraction)))
        n_val = min(n_val, len(shuffled) - 1)
        val_sessions = set(shuffled[:n_val])
        val_mask = np.isin(dataset.sessions, list(val_sessions))
    else:
        log.warning(
            "Hanya 1 sesi. Validasi memakai split acak frame; ini kurang "
            "mengukur generalisasi. Rekam sesi kedua untuk hasil terpercaya."
        )
        val_mask = np.zeros(len(y), dtype=bool)
        n_val = max(1, int(round(len(y) * val_fraction)))
        val_mask[rng.choice(len(y), size=n_val, replace=False)] = True
        val_sessions = set(sessions)

    train_mask = ~val_mask
    if not train_mask.any() or not val_mask.any():
        raise ValueError("Split data gagal: train/validasi kosong")
    if len(sessions) == 1:
        # Label sesi validasi tetap sesi tersebut; report tidak menyesatkan.
        val_sessions = set(sessions)

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

    model = _mlp(torch, FEATURE_DIM, n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight_t)
    best_loss = float("inf")
    best_state = None
    bad = 0
    epochs_run = 0
    n_train = int(train_mask.sum())

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
                "epoch %3d  val_loss %.4f  val_acc %.3f%s",
                epoch,
                val_loss,
                val_acc,
                "  *" if bad == 0 else "",
            )
        if bad >= patience:
            log.info("Early stop di epoch %d", epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_pred = model(xtr).argmax(dim=1)
        val_pred = model(xval).argmax(dim=1)
        train_acc = float((train_pred == ytr).float().mean().item())
        val_acc = float((val_pred == yval).float().mean().item())

    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    y_true = yval.detach().cpu().numpy()
    y_pred = val_pred.detach().cpu().numpy()
    for true, pred in zip(y_true, y_pred, strict=True):
        confusion[int(true), int(pred)] += 1

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    np.savez(
        out_path,
        W1=state["fc1.weight"].detach().cpu().numpy().T,
        b1=state["fc1.bias"].detach().cpu().numpy(),
        W2=state["fc2.weight"].detach().cpu().numpy().T,
        b2=state["fc2.bias"].detach().cpu().numpy(),
        W3=state["fc3.weight"].detach().cpu().numpy().T,
        b3=state["fc3.bias"].detach().cpu().numpy(),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        classes=np.array(dataset.classes, dtype=str),
    )
    return TrainingReport(
        out_path=out_path,
        device=device,
        epochs_run=epochs_run,
        best_val_loss=best_loss,
        train_accuracy=train_acc,
        val_accuracy=val_acc,
        confusion=confusion,
        train_sessions=tuple(sorted(set(dataset.sessions[train_mask]))),
        val_sessions=tuple(sorted(val_sessions)),
        classes=dataset.classes,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_gesture",
        description="Latih classifier gestur dari data landmark JSONL.",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        dataset = load_dataset(args.data)
        report = train_model(
            dataset,
            args.out,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            seed=args.seed,
            val_fraction=args.val_fraction,
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
    print(f"  Akurasi train: {report.train_accuracy * 100:.1f}%")
    print(f"  Akurasi val  : {report.val_accuracy * 100:.1f}%")
    print(f"  Sesi train   : {', '.join(report.train_sessions)}")
    print(f"  Sesi val     : {', '.join(report.val_sessions)}")
    print("  Confusion (baris=asli, kolom=prediksi):")
    for i, name in enumerate(report.classes):
        row = " ".join(f"{v:5d}" for v in report.confusion[i])
        print(f"    {name:8s} {row}")
    print()
    print("  Aktifkan di config.json:")
    print('    "gestures": { "classifier": { "enabled": true } }')
    print("  Lalu jalankan ulang: run.bat")
    print("=" * 62)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
