"""Plain backtracking: first blank cell, values 1-9 in order."""
import time
from src.csp.model import PEERS, to_list, find_conflicts
from src.csp.stats import Stats


def solve_bt(grid, max_assignments=None):
    """Return (solution_list or None, Stats).
    None means no solution (or the optional assignment cap was hit)."""
    g = to_list(grid)
    stats = Stats()
    t0 = time.perf_counter()

    if find_conflicts(g):                      # impossible: clashing givens
        stats.time_s = time.perf_counter() - t0
        return None, stats

    blanks = [i for i in range(81) if g[i] == 0]

    def consistent(i, v):
        return all(g[j] != v for j in PEERS[i])

    class _Cap(Exception):
        pass

    def rec(k):
        if k == len(blanks):
            return True
        i = blanks[k]
        for v in range(1, 10):
            if consistent(i, v):
                stats.assignments += 1
                if max_assignments and stats.assignments > max_assignments:
                    raise _Cap
                g[i] = v
                if rec(k + 1):
                    return True
                g[i] = 0
                stats.backtracks += 1
        return False

    try:
        ok = rec(0)
    except _Cap:
        ok = False
    stats.time_s = time.perf_counter() - t0
    return (g if ok else None), stats