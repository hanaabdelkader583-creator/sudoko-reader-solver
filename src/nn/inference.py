"""
src/nn/inference.py
===================
Raw Sudoku image  ->  9x9 NumPy array (0 = empty cell).
"""
from __future__ import annotations
import argparse, logging
from dataclasses import dataclass
from pathlib import Path
import cv2, numpy as np, torch
from .model import DEFAULT_WEIGHTS_PATH, IMG_SIZE, SudokuDigitCNN, get_device, load_model

log = logging.getLogger("sudoku.inference")
GRID, CELL_PX, WARP_PX, DIGIT_BOX_PX = 9, 50, 450, 20
BORDER_MARGIN_FRAC, CENTER_WINDOW = 0.10, (0.25, 0.75)
MIN_DIGIT_AREA_FRAC, MIN_DIGIT_HEIGHT_FRAC, FRAGMENT_AREA_FRAC = 0.02, 0.30, 0.15
MIN_FRAME_SOLIDITY, MIN_GRID_HOLES, THRESH_C = 0.90, 16, 6

def load_gray(source: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(source, (str, Path)):
        buf = np.fromfile(str(source), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    arr = np.asarray(source)
    if arr.ndim == 2: return arr
    return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.shape[2] == 3 else cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)

def normalize_polarity(gray: np.ndarray, polarity: str = "auto") -> np.ndarray:
    h, w = gray.shape
    centre = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    invert = polarity == "light_on_dark" or (polarity == "auto" and np.median(centre) < 128)
    return cv2.bitwise_not(gray) if invert else gray

def binarize(gray: np.ndarray, block: int, c: int = THRESH_C) -> np.ndarray:
    block = max(3, block | 1)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, c)

def find_grid_quad(ink: np.ndarray, min_area_frac: float = 0.20) -> np.ndarray | None:
    h, w = ink.shape
    bridged = cv2.dilate(ink, np.ones((3, 3), np.uint8))
    contours, hierarchy = cv2.findContours(bridged, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None: return None
    parent = hierarchy[0][:, 3]
    n_holes = np.bincount(parent[parent >= 0], minlength=len(contours))
    min_area = min_area_frac * h * w
    outer = [(cv2.contourArea(c), i) for i, c in enumerate(contours) if parent[i] < 0 and cv2.contourArea(c) >= min_area]
    if not outer: return None
    lattices = [(area, i) for area, i in outer if n_holes[i] >= MIN_GRID_HOLES]
    _, best = max(lattices or outer)
    hull = cv2.convexHull(contours[best])
    if cv2.contourArea(contours[best]) < MIN_FRAME_SOLIDITY * cv2.contourArea(hull): return None
    perimeter = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            s = pts.sum(axis=1); d = np.diff(pts, axis=1).ravel()
            return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)
    return None

def warp_grid(gray: np.ndarray, quad: np.ndarray | None) -> np.ndarray:
    if quad is None: return cv2.resize(gray, (WARP_PX, WARP_PX), interpolation=cv2.INTER_AREA)
    dst = np.array([[0, 0], [WARP_PX - 1, 0], [WARP_PX - 1, WARP_PX - 1], [0, WARP_PX - 1]], dtype=np.float32)
    return cv2.warpPerspective(gray, cv2.getPerspectiveTransform(quad, dst), (WARP_PX, WARP_PX))

def split_cells(img: np.ndarray) -> np.ndarray:
    cell = img.shape[0] // GRID
    return img[: cell * GRID, : cell * GRID].reshape(GRID, cell, GRID, cell).swapaxes(1, 2)

def extract_digit(cell_ink: np.ndarray) -> np.ndarray | None:
    h, w = cell_ink.shape
    m = int(round(BORDER_MARGIN_FRAC * h))
    work = np.zeros_like(cell_ink)
    work[m : h - m, m : w - m] = cell_ink[m : h - m, m : w - m]
    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lo, hi = CENTER_WINDOW
    cands = [(cv2.contourArea(cnt), cnt) for cnt in contours if lo * w <= cv2.boundingRect(cnt)[0] + cv2.boundingRect(cnt)[2]/2 <= hi * w and lo * h <= cv2.boundingRect(cnt)[1] + cv2.boundingRect(cnt)[3]/2 <= hi * h]
    if not cands: return None
    largest_area = max(a for a, _ in cands)
    if largest_area < MIN_DIGIT_AREA_FRAC * h * w: return None
    parts = [c for a, c in cands if a >= FRAGMENT_AREA_FRAC * largest_area]
    x, y, bw, bh = cv2.boundingRect(np.vstack(parts))
    if bh < MIN_DIGIT_HEIGHT_FRAC * h: return None
    region = np.zeros_like(work)
    cv2.drawContours(region, parts, -1, 255, thickness=cv2.FILLED)
    glyph = cv2.bitwise_and(work, region)
    return glyph[y : y + bh, x : x + bw]

def to_mnist_canvas(glyph: np.ndarray) -> np.ndarray:
    gh, gw = glyph.shape
    scale = DIGIT_BOX_PX / max(gh, gw)
    new_w, new_h = max(1, round(gw * scale)), max(1, round(gh * scale))
    small = cv2.resize(glyph, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)
    y0, x0 = (IMG_SIZE - new_h) // 2, (IMG_SIZE - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = small
    mom = cv2.moments(canvas)
    if mom["m00"] > 0:
        cx, cy = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
        centre = (IMG_SIZE - 1) / 2
        shift = np.float32([[1, 0, centre - cx], [0, 1, centre - cy]])
        canvas = cv2.warpAffine(canvas, shift, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR, borderValue=0)
    return canvas

@dataclass(frozen=True, eq=False)
class ReadResult:
    grid: np.ndarray
    confidence: np.ndarray
    warped: np.ndarray
    quad: np.ndarray | None
    @property
    def grid_found(self) -> bool: return self.quad is not None

class SudokuReader:
    def __init__(self, weights_path: str | Path = DEFAULT_WEIGHTS_PATH, *, model: SudokuDigitCNN | None = None, device: str | torch.device | None = None, polarity: str = "auto") -> None:
        self.device = get_device(device)
        if model is None: model, _ = load_model(weights_path, self.device)
        self.model = model.to(self.device).eval()
        self.polarity = polarity

    def _classify(self, canvases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = torch.from_numpy(canvases).to(self.device, dtype=torch.float32).div_(255.0).unsqueeze(1)
        probs = self.model.predict_proba(x)
        conf, label = probs.max(dim=1)
        return label.cpu().numpy(), conf.cpu().numpy()

    def read(self, source: str | Path | np.ndarray, *, require_grid: bool = False) -> ReadResult:
        gray = normalize_polarity(load_gray(source), self.polarity)
        block = max(11, min(gray.shape) // 20)
        quad = find_grid_quad(binarize(gray, block))
        warped = warp_grid(gray, quad)
        ink = binarize(warped, block=CELL_PX // 2)
        cells = split_cells(ink)
        canvases, coords = [], []
        for r in range(GRID):
            for c in range(GRID):
                glyph = extract_digit(np.ascontiguousarray(cells[r, c]))
                if glyph is not None:
                    canvases.append(to_mnist_canvas(glyph))
                    coords.append((r, c))
        grid = np.zeros((GRID, GRID), dtype=np.int64)
        confidence = np.ones((GRID, GRID), dtype=np.float32)
        if canvases:
            labels, confs = self._classify(np.stack(canvases))
            for (r, c), lab, cf in zip(coords, labels, confs):
                grid[r, c], confidence[r, c] = int(lab), float(cf)
        return ReadResult(grid=grid, confidence=confidence, warped=warped, quad=quad)