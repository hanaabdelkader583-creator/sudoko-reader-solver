"""
src/integration/pipeline.py
===========================
Bridge between the vision system (``src.nn``) and the CSP / backtracking solver,
plus the hybrid tiles -> NN -> correction search -> CSP pipeline.
"""
from __future__ import annotations
import argparse, functools, logging, time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING
import numpy as np
from ..nn.model import DEFAULT_WEIGHTS_PATH
from ..search.corrections import SearchResult, search_corrections

if TYPE_CHECKING:
    from ..nn.inference import SudokuReader

log = logging.getLogger("sudoku.pipeline")
GRID, BOX = 9, 3
Cell = tuple[int, int]
SolverFn = Callable[[np.ndarray], Optional[np.ndarray]]
MIN_GIVENS_FOR_UNIQUE_SOLUTION = 17


class SudokuReadError(ValueError):
    pass


def _build_units() -> tuple[tuple[Cell, ...], ...]:
    rows = [tuple((r, c) for c in range(GRID)) for r in range(GRID)]
    cols = [tuple((r, c) for r in range(GRID)) for c in range(GRID)]
    boxes = [tuple((br * BOX + i, bc * BOX + j) for i in range(BOX) for j in range(BOX))
             for br in range(BOX) for bc in range(BOX)]
    return tuple(rows + cols + boxes)


UNITS = _build_units()


def find_conflicts(grid: np.ndarray) -> frozenset[Cell]:
    bad: set[Cell] = set()
    for unit in UNITS:
        seen: dict[int, list[Cell]] = {}
        for cell in unit:
            v = int(grid[cell])
            if v:
                seen.setdefault(v, []).append(cell)
        for cells in seen.values():
            if len(cells) > 1:
                bad.update(cells)
    return frozenset(bad)


def is_valid_solution(solution: np.ndarray, puzzle: np.ndarray) -> bool:
    if solution.shape != (GRID, GRID) or puzzle.shape != (GRID, GRID):
        return False
    given = puzzle > 0
    if not np.array_equal(solution[given], puzzle[given]):
        return False
    target = list(range(1, GRID + 1))
    return all(sorted(int(solution[cell]) for cell in unit) == target for unit in UNITS)


@dataclass(frozen=True, eq=False)
class SudokuPuzzle:
    grid: np.ndarray
    confidence: np.ndarray
    conflicts: frozenset[Cell]
    low_confidence: frozenset[Cell]
    grid_found: bool = True

    @property
    def is_consistent(self) -> bool:
        return not self.conflicts

    @property
    def n_givens(self) -> int:
        return int(np.count_nonzero(self.grid))

    def to_numpy(self) -> np.ndarray:
        return self.grid.copy()

    def givens(self) -> dict[Cell, int]:
        return {(r, c): int(self.grid[r, c]) for r in range(GRID) for c in range(GRID)
                if self.grid[r, c]}


def _make_puzzle(grid: np.ndarray, confidence: np.ndarray, min_confidence: float,
                 grid_found: bool = True) -> SudokuPuzzle:
    grid = np.array(grid, dtype=np.int64)
    confidence = np.array(confidence, dtype=np.float32)
    grid.setflags(write=False)
    confidence.setflags(write=False)
    low = frozenset((r, c) for r in range(GRID) for c in range(GRID)
                    if grid[r, c] and confidence[r, c] < min_confidence)
    return SudokuPuzzle(grid, confidence, find_conflicts(grid), low, grid_found)


def puzzle_from_array(grid: np.ndarray | list, min_confidence: float = 0.0) -> SudokuPuzzle:
    arr = np.asarray(grid, dtype=np.int64)
    return _make_puzzle(arr, np.ones((GRID, GRID), np.float32), min_confidence)


@dataclass(eq=False)
class SolveOutcome:
    puzzle: SudokuPuzzle
    solution: np.ndarray | None
    solved: bool
    solver_seconds: float


class SudokuPipeline:
    def __init__(self, weights_path: str | Path = DEFAULT_WEIGHTS_PATH, *,
                 min_confidence: float = 0.90, device: str | None = None,
                 polarity: str = "auto", reader: "SudokuReader | None" = None) -> None:
        if reader is None:
            from ..nn.inference import SudokuReader  # lazy: needs cv2
            reader = SudokuReader(weights_path, device=device, polarity=polarity)
        self.reader = reader
        self.min_confidence = min_confidence

    def read(self, image: str | Path | np.ndarray, *, strict: bool = False) -> SudokuPuzzle:
        result = self.reader.read(image)
        puzzle = _make_puzzle(result.grid, result.confidence, self.min_confidence,
                              result.grid_found)
        if strict and puzzle.conflicts:
            raise SudokuReadError("recognised grid breaks Sudoku rules")
        return puzzle

    def solve(self, image: str | Path | np.ndarray | SudokuPuzzle, solver: SolverFn, *,
              strict: bool = True) -> SolveOutcome:
        puzzle = image if isinstance(image, SudokuPuzzle) else self.read(image, strict=strict)
        t0 = time.perf_counter()
        raw = solver(puzzle.to_numpy())
        elapsed = time.perf_counter() - t0
        solution = None if raw is None else np.asarray(raw, dtype=np.int64)
        solved = solution is not None and is_valid_solution(solution, puzzle.grid)
        return SolveOutcome(puzzle, solution, solved, elapsed)


# ----------------------------------------------------------------------------
# Tiles -> NN -> correction search -> CSP
# ----------------------------------------------------------------------------
@dataclass(eq=False)
class TileSolveResult:
    status: str                       # "solved" | "unreadable"
    reason: str
    read_grid: np.ndarray             # raw NN argmax grid (0 = blank)
    corrected_grid: np.ndarray        # after digit corrections
    corrections: list                 # (row, col, read, new)
    solution: np.ndarray | None
    suspects: list                    # (row, col, read_digit, confidence), least confident first
    search: SearchResult | None = None
    csp_stats: object = None
    method: str = "astar"

    @property
    def solved(self) -> bool:
        return self.status == "solved"


def _suspects(probs: np.ndarray, blank: np.ndarray, n: int) -> list:
    rows = [(i // GRID, i % GRID, int(probs[i].argmax()) + 1, float(probs[i].max()))
            for i in range(GRID * GRID) if not blank[i]]
    rows.sort(key=lambda t: t[3])
    return rows[:n]


def solve_from_tiles(tiles, *, method: str = "astar", predict=None, solver=None,
                     max_expansions: int = 200_000, top_k: int = 8,
                     n_suspects: int = 5, temperature: float = 1.0) -> TileSolveResult:
    """NN -> conflicts + correction search -> CSP, with an 'unreadable' report."""
    if predict is None:
        from ..nn.interface import predict_cells as predict
    if solver is None:
        from ..csp.solver import solve as solver

    probs, blank = predict(tiles, temperature=temperature)
    probs = np.asarray(probs, dtype=np.float64).reshape(GRID * GRID, GRID)
    blank = np.asarray(blank, dtype=bool).reshape(GRID * GRID)
    read_grid = np.where(blank, 0, probs.argmax(axis=1) + 1).reshape(GRID, GRID).astype(np.int64)
    suspects = _suspects(probs, blank, n_suspects)

    search = search_corrections(probs, blank, method=method,
                                max_expansions=max_expansions, top_k=top_k)
    if not search.found:
        return TileSolveResult(
            "unreadable",
            "no consistent set of digit corrections found (search budget exhausted or no repair exists)",
            read_grid, read_grid.copy(), [], None, suspects, search, None, method)

    corrected = np.asarray(search.grid, dtype=np.int64).reshape(GRID, GRID)
    try:
        out = solver(corrected.copy())
    except (TypeError, ValueError, AssertionError):
        out = solver(corrected.flatten().tolist())
    raw = out[0] if isinstance(out, tuple) else out
    stats = out[1] if isinstance(out, tuple) and len(out) > 1 else None

    if raw is None:
        return TileSolveResult(
            "unreadable", "CSP found no solution for the corrected grid",
            read_grid, corrected, search.corrections, None, suspects, search, stats, method)

    solution = np.asarray(raw, dtype=np.int64).reshape(GRID, GRID)
    if not is_valid_solution(solution, corrected):
        return TileSolveResult(
            "unreadable", "CSP returned an invalid grid (no solution accepted)",
            read_grid, corrected, search.corrections, None, suspects, search, stats, method)

    return TileSolveResult("solved", "", read_grid, corrected, search.corrections,
                           solution, suspects, search, stats, method)
