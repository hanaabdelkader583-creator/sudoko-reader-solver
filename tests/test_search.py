"""Tests for src/search/corrections.py (8 tests)."""
import numpy as np

from src.search.corrections import find_conflicts, search_corrections

SOL = np.array([
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
])


def make_probs(grid, misreads=None, blanks=()):
    """One-hot-ish probs; misreads = {(r, c): wrong_digit} -> wrong 0.6, truth 0.35."""
    misreads = misreads or {}
    probs = np.full((81, 9), 0.01)
    blank = np.zeros(81, dtype=bool)
    for r in range(9):
        for c in range(9):
            i = r * 9 + c
            if (r, c) in blanks:
                blank[i], probs[i] = True, 0.0
            elif (r, c) in misreads:
                probs[i, misreads[(r, c)] - 1] = 0.60
                probs[i, grid[r, c] - 1] = 0.35
            else:
                probs[i, grid[r, c] - 1] = 0.92
    return probs, blank


def test_no_conflicts_needs_no_corrections():
    probs, blank = make_probs(SOL)
    for m in ("ucs", "greedy", "astar"):
        res = search_corrections(probs, blank, m)
        assert res.found and res.corrections == [] and res.cost == 0.0
        assert np.array_equal(res.grid, SOL)


def test_single_misread_is_repaired():
    probs, blank = make_probs(SOL, {(0, 0): 3})
    for m in ("ucs", "astar"):
        res = search_corrections(probs, blank, m)
        assert res.found and np.array_equal(res.grid, SOL)
        assert res.corrections == [(0, 0, 3, 5)]


def test_ucs_and_astar_have_equal_optimal_cost():
    probs, blank = make_probs(SOL, {(0, 0): 3, (4, 4): 8})
    u = search_corrections(probs, blank, "ucs")
    a = search_corrections(probs, blank, "astar")
    assert u.found and a.found
    assert abs(u.cost - a.cost) < 1e-9


def test_astar_expands_no_more_than_ucs():
    probs, blank = make_probs(SOL, {(0, 0): 3, (4, 4): 8, (8, 8): 1})
    u = search_corrections(probs, blank, "ucs")
    a = search_corrections(probs, blank, "astar")
    assert a.expanded <= u.expanded


def test_greedy_returns_conflict_free_grid():
    probs, blank = make_probs(SOL, {(0, 0): 3, (4, 4): 8})
    res = search_corrections(probs, blank, "greedy")
    assert res.found and find_conflicts(res.grid) == []


def test_blank_cells_are_never_touched():
    blanks = {(1, 1), (2, 5), (7, 7)}
    probs, blank = make_probs(SOL, {(0, 0): 3}, blanks)
    res = search_corrections(probs, blank, "astar")
    assert res.found
    for r, c in blanks:
        assert res.grid[r, c] == 0
    assert all((r, c) not in blanks for r, c, _, _ in res.corrections)


def test_find_conflicts_detects_row_col_box():
    g = np.zeros((9, 9), dtype=int)
    g[0, 0] = g[0, 5] = 4      # row
    g[3, 8] = g[7, 8] = 2      # column
    g[3, 0] = g[4, 1] = 9      # box
    assert len(find_conflicts(g)) == 3
    assert find_conflicts(SOL) == []


def test_expansion_limit_reports_failure():
    probs, blank = make_probs(SOL, {(0, 0): 3, (4, 4): 8})
    res = search_corrections(probs, blank, "ucs", max_expansions=1)
    assert not res.found