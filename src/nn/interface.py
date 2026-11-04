"""NN inference contract: 81 tiles -> (probs (81, 9), blank (81,))."""
import numpy as np
import torch
from src.nn.model import load_model, DEFAULT_WEIGHTS_PATH
from src.data.mosaic import blank_mask

_model = None


def _get_model():
    global _model
    if _model is None:
        if not DEFAULT_WEIGHTS_PATH.is_file():
            raise FileNotFoundError(f"weights not found: {DEFAULT_WEIGHTS_PATH}")
        _model = load_model(DEFAULT_WEIGHTS_PATH)
    return _model


def predict_cells(tiles, temperature=1.0):
    """tiles: (81, 28, 28) float32 in 0-1.
    Returns probs (81, 9) over digits 1-9 (zeros on blank cells) and blank (81,) bool."""
    tiles = np.asarray(tiles, dtype=np.float32)
    assert tiles.shape == (81, 28, 28)
    blank = blank_mask(tiles)
    model = _get_model()
    device = next(model.parameters()).device
    with torch.no_grad():
        x = torch.from_numpy(tiles).unsqueeze(1).to(device)
        logits = model(x)[:, 1:10] / temperature     # classes 1-9 only
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    probs[blank] = 0.0
    return probs, blank