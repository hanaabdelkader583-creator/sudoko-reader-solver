"""Tests for src/integration/pipeline.py (6 tests, NN and CSP injected)."""
import numpy as np

from src.integration.pipeline import solve_from_tiles
from tests.test_search import SOL, make_probs

TILES = np.zeros((81, 28, 28), dtype=np.float32)


def fake_predict(probs, blank):
    return lambda tiles, **kw: (probs.copy(), blank.copy())


def fake_solver(grid, **kw):
    """Stand-in CSP: 'solves' by returning SOL if the grid agrees with it."""
    g = np.asarray(grid)
    ok = np.all((g == 0) | (g == SOL))
    return (SOL.flatten().tolist() if ok else None), None


def test_clean_read_is_solved_without_corrections():
    probs, blank = make_probs(SOL, blanks={(0, 2), (5, 5)})
    r = solve_from_tiles(TILES, predict=fake_predict(probs, blank), solver=fake_solver)
    assert r.solved and r.corrections == []
    assert np.array_equal(r.solution, SOL)


def test_single_misread_is_corrected_then_solved():
    probs, blank = make_probs(SOL, {(0, 0): 3})
    r = solve_from_tiles(TILES, predict=fake_predict(probs, blank), solver=fake_solver)
    assert r.solved and r.corrections == [(0, 0, 3, 5)]
    assert r.read_grid[0, 0] == 3 and r.corrected_grid[0, 0] == 5


def test_all_three_search_methods_give_a_solution():
    probs, blank = make_probs(SOL, {(0, 0): 3, (4, 4): 8})
    for m in ("ucs", "greedy", "astar"):
        r = solve_from_tiles(TILES, method=m, predict=fake_predict(probs, blank), solver=fake_solver)
        assert r.solved and np.array_equal(r.solution, SOL)


def test_unrepairable_conflicts_are_reported_unreadable():
    probs, blank = make_probs(SOL, {(0, 0): 3})
    r = solve_from_tiles(TILES, predict=fake_predict(probs, blank), solver=fake_solver,
                         max_expansions=0)
    assert r.status == "unreadable" and r.suspects
    assert r.solution is None


def test_unsolvable_corrected_grid_is_reported_with_suspects():
    # (0,0) is read as 3 and the two cells that would conflict with it are blank,
    # so the misread is invisible to the conflict check; the CSP must reject it.
    probs, blank = make_probs(SOL, {(0, 0): 3}, blanks={(0, 1), (8, 0)})
    r = solve_from_tiles(TILES, predict=fake_predict(probs, blank), solver=fake_solver)
    assert r.status == "unreadable" and "no solution" in r.reason
    assert r.suspects[0][3] <= r.suspects[-1][3]
    assert (0, 0) == r.suspects[0][:2]


def test_real_csp_solves_corrected_grid():
    from src.csp.solver import solve
    blanks = {(r, c) for r in range(9) for c in range(9) if (r * 3 + c) % 3 == 0}
    probs, blank = make_probs(SOL, {(0, 1): 4}, blanks)
    r = solve_from_tiles(TILES, predict=fake_predict(probs, blank), solver=solve)
    assert r.solved and np.array_equal(r.solution, SOL)
    