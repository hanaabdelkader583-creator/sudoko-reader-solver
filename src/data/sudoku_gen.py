"""Seeded Sudoku generator with an independent solution counter.
Grids are flat length-81 arrays, 0 = blank. Deliberately independent of src/csp."""
import numpy as np
import config

ROW = [i // 9 for i in range(81)]
COL = [i % 9 for i in range(81)]
BOX = [(i // 27) * 3 + (i % 9) // 3 for i in range(81)]
ALL_BITS = 0x3FE  # bits 1..9


def count_solutions(grid, limit=2):
    """Count solutions, stopping at `limit`. Returns 0 if givens already clash."""
    g = [int(v) for v in grid]
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    for i, v in enumerate(g):
        if v:
            b = 1 << v
            if rows[ROW[i]] & b or cols[COL[i]] & b or boxes[BOX[i]] & b:
                return 0
            rows[ROW[i]] |= b
            cols[COL[i]] |= b
            boxes[BOX[i]] |= b
    count = 0

    def rec():
        nonlocal count
        best, best_mask, best_n = -1, 0, 10
        for i in range(81):
            if g[i] == 0:
                avail = ~(rows[ROW[i]] | cols[COL[i]] | boxes[BOX[i]]) & ALL_BITS
                n = bin(avail).count("1")
                if n == 0:
                    return
                if n < best_n:
                    best, best_mask, best_n = i, avail, n
        if best == -1:
            count += 1
            return
        for v in range(1, 10):
            if (best_mask >> v) & 1:
                b = 1 << v
                g[best] = v
                rows[ROW[best]] |= b; cols[COL[best]] |= b; boxes[BOX[best]] |= b
                rec()
                g[best] = 0
                rows[ROW[best]] &= ~b; cols[COL[best]] &= ~b; boxes[BOX[best]] &= ~b
                if count >= limit:
                    return

    rec()
    return count


def generate_full_grid(rng):
    """Randomised backtracking: cells in order, digits in shuffled order."""
    g = [0] * 81
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9

    def rec(i):
        if i == 81:
            return True
        for v in rng.permutation(np.arange(1, 10)).tolist():
            b = 1 << v
            if rows[ROW[i]] & b or cols[COL[i]] & b or boxes[BOX[i]] & b:
                continue
            g[i] = v
            rows[ROW[i]] |= b; cols[COL[i]] |= b; boxes[BOX[i]] |= b
            if rec(i + 1):
                return True
            g[i] = 0
            rows[ROW[i]] &= ~b; cols[COL[i]] &= ~b; boxes[BOX[i]] &= ~b
        return False

    assert rec(0)
    return np.array(g, dtype=np.int8)


def make_puzzle(n_givens, seed):
    """Return (puzzle, solution) with exactly n_givens givens and a UNIQUE solution.
    Random-order removal can get stuck above the target, so we retry with a new
    sub-seed derived from (seed, attempt); still fully reproducible."""
    for attempt in range(config.MAX_GEN_ATTEMPTS):
        rng = np.random.default_rng([seed, attempt])
        solution = generate_full_grid(rng)
        puzzle = solution.copy()
        givens = 81
        for cell in rng.permutation(81):
            if givens == n_givens:
                break
            saved = puzzle[cell]
            puzzle[cell] = 0
            if count_solutions(puzzle, 2) == 1:
                givens -= 1
            else:
                puzzle[cell] = saved
        if givens == n_givens:
            return puzzle, solution
    raise RuntimeError(f"could not reach {n_givens} givens for seed {seed}")