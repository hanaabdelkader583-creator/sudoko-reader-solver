"""Read MNIST IDX files with numpy (no hidden transforms), split, drop digit 0."""
import gzip
import struct
import numpy as np
import config

FILES = {
    "train_images": "train-images-idx3-ubyte",
    "train_labels": "train-labels-idx1-ubyte",
    "test_images": "t10k-images-idx3-ubyte",
    "test_labels": "t10k-labels-idx1-ubyte",
}


def _open(name):
    plain = config.RAW_DIR / name
    gz = config.RAW_DIR / (name + ".gz")
    if plain.exists():
        return open(plain, "rb")
    if gz.exists():
        return gzip.open(gz, "rb")
    raise FileNotFoundError(name)


def _have_all():
    return all((config.RAW_DIR / n).exists() or (config.RAW_DIR / (n + ".gz")).exists()
               for n in FILES.values())


def ensure_mnist():
    """Use the local copy; download via torchvision only if raw/ is incomplete."""
    if _have_all():
        return
    from torchvision import datasets
    datasets.MNIST(root=str(config.DATA_DIR), train=True, download=True)
    datasets.MNIST(root=str(config.DATA_DIR), train=False, download=True)


def read_images(name):
    with _open(name) as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, "bad image file"
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)


def read_labels(name):
    with _open(name) as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, "bad label file"
        return np.frombuffer(f.read(), dtype=np.uint8)


def to_float(images_uint8):
    """Scale pixels to 0-1 float32 (images are stored as uint8 on disk)."""
    return images_uint8.astype(np.float32) / 255.0


def _drop_zero(images, labels, orig_idx):
    keep = labels != 0
    return images[keep], labels[keep], orig_idx[keep]


def load_mnist_split(seed):
    """50k train / 10k val from the training set (seeded), test untouched.
    Digit 0 is removed AFTER splitting. Labels stay 1-9."""
    ensure_mnist()
    tr_x = read_images(FILES["train_images"])
    tr_y = read_labels(FILES["train_labels"])
    te_x = read_images(FILES["test_images"])
    te_y = read_labels(FILES["test_labels"])
    assert len(tr_x) == 60_000 and len(te_x) == 10_000

    perm = np.random.default_rng(seed).permutation(len(tr_x))
    train_idx = perm[:config.N_TRAIN]
    val_idx = perm[config.N_TRAIN:config.N_TRAIN + config.N_VAL]
    test_idx = np.arange(len(te_x))

    out = {}
    for name, x, y, idx in [("train", tr_x, tr_y, train_idx),
                            ("val", tr_x, tr_y, val_idx),
                            ("test", te_x, te_y, test_idx)]:
        xi, yi, oi = _drop_zero(x[idx], y[idx], idx)
        out[f"{name}_images"] = xi
        out[f"{name}_labels"] = yi
        out[f"{name}_orig_idx"] = oi
    return out