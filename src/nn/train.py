"""
src/nn/train.py
===============
Train ``SudokuDigitCNN`` and save the weights.
"""
from __future__ import annotations
import argparse, gzip, logging, random, struct, time
from pathlib import Path
import cv2, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, get_worker_info
from .model import (BLANK_CLASS, DEFAULT_WEIGHTS_PATH, IMG_SIZE, NUM_CLASSES, PROJECT_ROOT, SudokuDigitCNN, count_parameters, get_device, load_model, save_checkpoint)

log = logging.getLogger("sudoku.train")
DEFAULT_NPZ = PROJECT_ROOT / "data" / "processed" / "mnist_split.npz"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "MNIST" / "raw"
Split = tuple[np.ndarray, np.ndarray]

def _as_uint8_images(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2 and a.shape[1] == IMG_SIZE * IMG_SIZE:
        a = a.reshape(-1, IMG_SIZE, IMG_SIZE)
    elif a.ndim == 4 and a.shape[1] == 1:
        a = a[:, 0]
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        if a.max() <= 1.0: a = a * 255.0
        a = np.clip(np.rint(a), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)

def _read_idx(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if path.suffix == ".gz": raw = gzip.decompress(raw)
    magic = struct.unpack(">I", raw[:4])[0]
    ndim = magic & 0xFF
    dims = struct.unpack(">" + "I" * ndim, raw[4 : 4 + 4 * ndim])
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)

def _carve_validation(images: np.ndarray, labels: np.ndarray, frac: float, seed: int) -> tuple[Split, Split]:
    perm = np.random.default_rng(seed).permutation(len(images))
    n_val = int(round(len(images) * frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return (images[train_idx], labels[train_idx]), (images[val_idx], labels[val_idx])

def load_raw_mnist(raw_dir: Path, val_fraction: float, seed: int) -> dict[str, Split]:
    def get(stem: str) -> np.ndarray:
        for name in (stem, stem + ".gz"):
            if (raw_dir / name).is_file(): return _read_idx(raw_dir / name)
        raise FileNotFoundError(f"{stem} not found in {raw_dir}")
    x_tr, y_tr = get("train-images-idx3-ubyte"), get("train-labels-idx1-ubyte")
    x_te, y_te = get("t10k-images-idx3-ubyte"), get("t10k-labels-idx1-ubyte")
    train, val = _carve_validation(_as_uint8_images(x_tr), y_tr.astype(np.int64), val_fraction, seed)
    return {"train": train, "val": val, "test": (_as_uint8_images(x_te), y_te.astype(np.int64))}

_SPLIT_ALIASES = {"train": ("train",), "val": ("val", "valid", "validation"), "test": ("test",)}
_SPLIT_KEY_PATTERNS = {
    "images": ("x_{s}", "{s}_x", "{s}_images", "images_{s}", "{s}_imgs", "{s}_data", "{s}_X"),
    "labels": ("y_{s}", "{s}_y", "{s}_labels", "labels_{s}", "{s}_targets", "{s}_Y"),
}

def load_processed_npz(npz_path: Path, val_fraction: float, seed: int) -> dict[str, Split]:
    with np.load(npz_path, allow_pickle=False) as npz:
        keys = {k.lower(): k for k in npz.files}
        def find(kind: str, split: str) -> str | None:
            for alias in _SPLIT_ALIASES[split]:
                for pattern in _SPLIT_KEY_PATTERNS[kind]:
                    name = pattern.format(s=alias).lower()
                    if name in keys: return keys[name]
            return None
        out: dict[str, Split] = {}
        for split in _SPLIT_ALIASES:
            ki, kl = find("images", split), find("labels", split)
            if ki and kl: out[split] = (_as_uint8_images(npz[ki]), npz[kl].astype(np.int64).ravel())
    if "train" not in out: raise KeyError("No train images/labels found.")
    if "val" not in out: out["train"], out["val"] = _carve_validation(*out["train"], val_fraction, seed)
    return out

def load_splits(source: str, npz_path: Path, raw_dir: Path, val_fraction: float, seed: int) -> dict[str, Split]:
    if source in ("npz", "auto"):
        try:
            return load_processed_npz(npz_path, val_fraction, seed)
        except Exception:
            if source == "npz": raise
    return load_raw_mnist(raw_dir, val_fraction, seed)

def _augment_digit(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    angle = rng.uniform(-12, 12)
    scale = rng.uniform(0.85, 1.15)
    tx, ty = rng.uniform(-2, 2, size=2)
    m = cv2.getRotationMatrix2D(((IMG_SIZE - 1) / 2, (IMG_SIZE - 1) / 2), angle, scale)
    m[:, 2] += (tx, ty)
    out = cv2.warpAffine(img, m, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR, borderValue=0)
    r = rng.random()
    if r < 0.30: out = cv2.dilate(out, np.ones((2, 2), np.uint8))
    elif r < 0.45: out = cv2.erode(out, np.ones((2, 2), np.uint8))
    if rng.random() < 0.5:
        noise = rng.normal(0.0, rng.uniform(0, 12), out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out

def _synth_blank(rng: np.random.Generator) -> np.ndarray:
    img = rng.normal(0.0, rng.uniform(0, 10), (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    for _ in range(int(rng.integers(0, 4))):
        centre = (int(rng.integers(0, IMG_SIZE)), int(rng.integers(0, IMG_SIZE)))
        cv2.circle(img, centre, int(rng.integers(1, 3)), float(rng.uniform(60, 255)), -1)
    return np.clip(img, 0, 255).astype(np.uint8)

class SudokuDigitDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, n_blank: int, augment: bool, seed: int = 0) -> None:
        self.images, self.labels = images, labels
        self.n_real, self.n_blank = len(images), n_blank
        self.augment, self.seed = augment, seed
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int: return self.n_real + self.n_blank

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rng = self._rng if self.augment else np.random.default_rng([self.seed, idx])
        if idx < self.n_real:
            img, label = self.images[idx], int(self.labels[idx])
            if self.augment: img = _augment_digit(img, rng)
        else:
            img, label = _synth_blank(rng), BLANK_CLASS
        x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        return x, label

def _seed_worker(worker_id: int) -> None:
    info = get_worker_info()
    info.dataset._rng = np.random.default_rng(info.seed)

def build_dataset(split: Split, augment: bool, blank_ratio: float, seed: int) -> SudokuDigitDataset:
    images, labels = split
    keep = labels != 0
    images, labels = images[keep], labels[keep]
    n_blank = int(round(len(images) / 9 * blank_ratio))
    return SudokuDigitDataset(images, labels, n_blank, augment, seed)

def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            seen += x.size(0)
    return total_loss / seen, correct / seen

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=("auto", "npz", "raw"), default="auto")
    p.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--blank-ratio", type=float, default=1.0)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cv2.setNumThreads(0)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = get_device(args.device)

    splits = load_splits(args.source, args.npz, args.raw_dir, args.val_fraction, args.seed)
    train_ds = build_dataset(splits["train"], augment=True, blank_ratio=args.blank_ratio, seed=args.seed)
    val_ds = build_dataset(splits["val"], augment=False, blank_ratio=args.blank_ratio, seed=args.seed + 1)
    
    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda", worker_init_fn=_seed_worker if args.workers else None)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(args.seed), **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, **loader_kw)

    model = SudokuDigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer, scheduler)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        if va_acc > best_acc:
            best_acc = va_acc
            save_checkpoint(args.weights, model, meta={"epoch": epoch, "val_acc": va_acc})

if __name__ == "__main__":
    main()