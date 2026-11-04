"""
src/integration/pipeline.py
===========================
Bridge between the vision system (``src.nn``) and the CSP / backtracking solver.
"""
from __future__ import annotations
import argparse, functools, logging, time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
import numpy as np
from ..nn.inference import SudokuReader
from ..nn.model import DEFAULT_WEIGHTS_PATH

log = logging.getLogger("sudoku.pipeline")
GRID, BOX = 9, 3
Cell = tuple[int, int]
SolverFn = Callable[[np.ndarray], Optional[np.ndarray]]
MIN_GIVENS_FOR_UNIQUE_SOLUTION = 17

class SudokuReadError(ValueError): pass

def _build_units() -> tuple[tuple[Cell, ...], ...]:
    rows = [tuple((r, c) for c in range(GRID)) for r in range(GRID)]
    cols = [tuple((r, c) for r in range(GRID)) for c in range(GRID)]
    boxes = [tuple((br * BOX + i, bc * BOX + j) for i in range(BOX) for j in range(BOX)) for br in range(BOX) for bc in range(BOX)]
    return tuple(rows + cols + boxes)
UNITS = _build_units()

def find_conflicts(grid: np.ndarray) -> frozenset[Cell]:
    bad: set[Cell] = set()
    for unit in UNITS:
        seen: dict[int, list[Cell]] = {}
        for cell in unit:
            v = int(grid[cell])
            if v: seen.setdefault(v, []).append(cell)
        for cells in seen.values():
            if len(cells) > 1: bad.update(cells)
    return frozenset(bad)

def is_valid_solution(solution: np.ndarray, puzzle: np.ndarray) -> bool:
    if solution.shape != (GRID, GRID) or puzzle.shape != (GRID, GRID): return False
    given = puzzle > 0
    if not np.array_equal(solution[given], puzzle[given]): return False
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
    def is_consistent(self) -> bool: return not self.conflicts
    @property
    def n_givens(self) -> int: return int(np.count_nonzero(self.grid))
    def to_numpy(self) -> np.ndarray: return self.grid.copy()
    def givens(self) -> dict[Cell, int]: return {(r, c): int(self.grid[r, c]) for r in range(GRID) for c in range(GRID) if self.grid[r, c]}

def _make_puzzle(grid: np.ndarray, confidence: np.ndarray, min_confidence: float, grid_found: bool = True) -> SudokuPuzzle:
    grid = np.array(grid, dtype=np.int64)
    confidence = np.array(confidence, dtype=np.float32)
    grid.setflags(write=False); confidence.setflags(write=False)
    low = frozenset((r, c) for r in range(GRID) for c in range(GRID) if grid[r, c] and confidence[r, c] < min_confidence)
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
    def __init__(self, weights_path: str | Path = DEFAULT_WEIGHTS_PATH, *, min_confidence: float = 0.90, device: str | None = None, polarity: str = "auto", reader: SudokuReader | None = None) -> None:
        self.reader = reader or SudokuReader(weights_path, device=device, polarity=polarity)
        self.min_confidence = min_confidence

    def read(self, image: str | Path | np.ndarray, *, strict: bool = False) -> SudokuPuzzle:
        result = self.reader.read(image)
        puzzle = _make_puzzle(result.grid, result.confidence, self.min_confidence, result.grid_found)
        if strict and puzzle.conflicts: raise SudokuReadError("recognised grid breaks Sudoku rules")
        return puzzle

    def solve(self, image: str | Path | np.ndarray | SudokuPuzzle, solver: SolverFn, *, strict: bool = True) -> SolveOutcome:
        puzzle = image if isinstance(image, SudokuPuzzle) else self.read(image, strict=strict)
        t0 = time.perf_counter()
        raw = solver(puzzle.to_numpy())
        elapsed = time.perf_counter() - t0
        solution = None if raw is None else np.asarray(raw, dtype=np.int64)
        solved = solution is not None and is_valid_solution(solution, puzzle.grid)
        return SolveOutcome(puzzle, solution, solved, elapsed)