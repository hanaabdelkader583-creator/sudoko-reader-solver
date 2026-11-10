"""
src/eval/correction_benchmark.py
================================
Shared machinery for Experiments 1 and 3 and the demo: build puzzles, plant real
NN misreads in them, and run the correction pipelines.
"""
from __future__ import annotations
import csv
import time
import numpy as np
import config
from src.data.sudoku_gen import make_puzzle
from src.data.mosaic import load_raw_test
from src.data.misreads import MisreadPool, make_misread_instance
from src.integration.pipeline import solve_from_tiles, find_conflicts, is_valid_solution

N_GIVENS = config.GIVENS_LEVELS[1]       # 30 givens
MISREAD_LEVELS = (0, 1, 2, 3, 4)
MAX_CORRECTIONS = 3                      # cap from the project brief
MAX_EXPANSIONS = 300
UNIQUE_GOAL = True
RANDOM_TRIES = 200


def csp_solve(grid, **kw):
    """Real CSP. Takes a 9x9 or flat grid, returns a 9x9 array or None."""
    from src.csp.solver import solve
    flat = np.asarray(grid, dtype=np.int64).flatten().tolist()
    out = solve(flat, **kw)
    raw = out[0] if isinstance(out, tuple) else out
    return None if raw is None else np.asarray(raw, dtype=np.int64).reshape(9, 9)


def puzzle_seed(group, j):
    return config.SEED * 1000 + group * 100 + j


def load_world():
    imgs, labels = load_raw_test()
    return imgs, MisreadPool.from_model(imgs, labels)


def read_grid_from(probs, blank):
    return np.where(blank, 0, probs.argmax(axis=1) + 1).reshape(9, 9).astype(np.int64)


def iter_instances(n_per_group=None, levels=MISREAD_LEVELS):
    """Yield one dict per (puzzle, misread level). Same puzzles are reused at every level."""
    from src.nn.interface import predict_cells
    n_per_group = n_per_group or config.PUZZLES_PER_GROUP
    imgs, pool = load_world()
    for g in range(config.N_SEED_GROUPS):
        for j in range(n_per_group):
            ps = puzzle_seed(g, j)
            puzzle, truth = make_puzzle(N_GIVENS, ps)
            puzzle = np.asarray(puzzle).reshape(81)
            truth = np.asarray(truth).reshape(9, 9)
            for k in levels:
                base = dict(group=g, puzzle=j, level=k)
                try:
                    tiles, injected, _ = make_misread_instance(puzzle, k, ps * 10 + k, pool, imgs)
                except ValueError:
                    yield dict(base, skipped=True)
                    continue
                probs, blank = predict_cells(tiles)
                yield dict(base, skipped=False, truth=truth, tiles=tiles, injected=injected,
                           probs=probs, blank=blank, read_grid=read_grid_from(probs, blank),
                           seed=ps * 10 + k)


def fmt_corr(corr):
    return ";".join(f"r{r}c{c}:{a}->{b}" for r, c, a, b in corr)


def _record(inst, method, corrections=(), **kw):
    rec = dict(group=inst["group"], puzzle=inst["puzzle"], level=inst["level"], method=method,
               corrections=fmt_corr(corrections),
               injected=";".join(f"r{r}c{c}:{t}->{g}" for (r, c), (t, g) in inst["injected"].items()))
    rec.update(kw)
    return rec


def run_search(inst, method, max_corrections=MAX_CORRECTIONS, max_expansions=MAX_EXPANSIONS,
               solvable_goal=True, unique_goal=UNIQUE_GOAL):
    """NN probabilities -> correction search (ucs/greedy/astar) -> real CSP."""
    probs, blank = inst["probs"], inst["blank"]
    t0 = time.perf_counter()
    r = solve_from_tiles(inst["tiles"], method=method,
                         predict=lambda t, **kw: (probs.copy(), blank.copy()),
                         solver=csp_solve, max_expansions=max_expansions, top_k=3,
                         max_corrections=max_corrections, solvable_goal=solvable_goal,
                         unique_goal=unique_goal)
    dt = time.perf_counter() - t0
    recovered = bool(r.solved and np.array_equal(r.solution, inst["truth"]))
    return _record(inst, method, r.corrections, solved=float(r.solved), recovered=float(recovered),
                   expanded=r.search.expanded, generated=r.search.generated,
                   n_corr=len(r.corrections),
                   cost=(r.search.cost if r.search.found else float("nan")),
                   time_s=dt, status=r.status, reason=r.reason)


def run_none(inst):
    """No correction: hand the raw NN reading straight to the CSP."""
    t0 = time.perf_counter()
    sol = csp_solve(inst["read_grid"])
    dt = time.perf_counter() - t0
    solved = sol is not None and is_valid_solution(sol, inst["read_grid"])
    recovered = bool(solved and np.array_equal(sol, inst["truth"]))
    return _record(inst, "none", solved=float(solved), recovered=float(recovered), expanded=0,
                   generated=0, n_corr=0, cost=0.0, time_s=dt,
                   status="solved" if solved else "unreadable", reason="")


def run_random(inst, tries=RANDOM_TRIES, max_corr=MAX_CORRECTIONS):
    """Random baseline: up to `tries` attempts, each changing 1..3 random conflicting
    cells to random digits; the first conflict-free grid goes to the CSP."""
    rng = np.random.default_rng(inst["seed"])
    read = inst["read_grid"]
    t0 = time.perf_counter()
    grid, corr, used = read, [], 0
    if find_conflicts(read):
        grid = None
        for used in range(1, tries + 1):
            g, corr = read.copy(), []
            for _ in range(int(rng.integers(1, max_corr + 1))):
                cells = sorted(find_conflicts(g))
                if not cells:
                    break
                r, c = cells[int(rng.integers(len(cells)))]
                new = int(rng.choice([d for d in range(1, 10) if d != g[r, c]]))
                corr.append((r, c, int(g[r, c]), new))
                g[r, c] = new
            if not find_conflicts(g):
                grid = g
                break
    sol = None if grid is None else csp_solve(grid)
    dt = time.perf_counter() - t0
    solved = sol is not None and is_valid_solution(sol, grid)
    recovered = bool(solved and np.array_equal(sol, inst["truth"]))
    return _record(inst, "random", corr if grid is not None else [], solved=float(solved),
                   recovered=float(recovered), expanded=0, generated=used, n_corr=len(corr),
                   cost=float("nan"), time_s=dt,
                   status="solved" if solved else "unreadable", reason="")


def aggregate(records, metric):
    """{(level, method): (mean, std)}; std is the sample std across the seed groups."""
    cells = {}
    for r in records:
        cells.setdefault((r["level"], r["method"], r["group"]), []).append(r[metric])
    per = {}
    for (lv, m, g), vals in cells.items():
        per.setdefault((lv, m), []).append(float(np.mean(np.asarray(vals, dtype=float))))
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
            for k, v in per.items()}


def pm(ms, scale=1.0, nd=1):
    return f"{ms[0] * scale:.{nd}f} \u00b1 {ms[1] * scale:.{nd}f}"


def write_log(records, path):
    cols = ["group", "puzzle", "level", "method", "status", "solved", "recovered", "cost",
            "n_corr", "corrections", "injected", "reason"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in records:
            w.writerow([r.get(c, "") for c in cols])
