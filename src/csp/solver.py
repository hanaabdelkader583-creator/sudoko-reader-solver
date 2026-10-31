"""Switchable CSP solver: BT, BT+FC, (+MRV). AC-3 is added in the next step."""
import time
from src.csp.model import PEERS, to_list, find_conflicts, initial_domains
from src.csp.stats import Stats


class _Cap(Exception):
    pass


def solve(grid, use_fc=True, use_mrv=True, max_assignments=None):
    """Return (solution_list or None, Stats)."""
    g = to_list(grid)
    stats = Stats()
    t0 = time.perf_counter()

    if find_conflicts(g):
        stats.time_s = time.perf_counter() - t0
        return None, stats

    dom = initial_domains(g)

    # Givens are already placed: remove their digit from their peers' domains.
    if use_fc:
        for i in range(81):
            if g[i]:
                for j in PEERS[i]:
                    if g[j] == 0 and g[i] in dom[j]:
                        dom[j].discard(g[i])
                        stats.prunings += 1
                if any(g[j] == 0 and not dom[j] for j in PEERS[i]):
                    stats.time_s = time.perf_counter() - t0
                    return None, stats

    def pick():
        blanks = [i for i in range(81) if g[i] == 0]
        if not blanks:
            return None
        if use_mrv:
            return min(blanks, key=lambda i: len(dom[i]))
        return blanks[0]

    def values(i):
        if use_fc:
            return sorted(dom[i])
        return [v for v in range(1, 10) if all(g[j] != v for j in PEERS[i])]

    def rec():
        i = pick()
        if i is None:
            return True
        for v in values(i):
            stats.assignments += 1
            if max_assignments and stats.assignments > max_assignments:
                raise _Cap
            g[i] = v
            removed = []
            ok = True
            if use_fc:
                for j in PEERS[i]:
                    if g[j] == 0 and v in dom[j]:
                        dom[j].discard(v)
                        removed.append(j)
                        stats.prunings += 1
                        if not dom[j]:
                            ok = False
                            break
            if ok and rec():
                return True
            for j in removed:                  # undo FC pruning
                dom[j].add(v)
            g[i] = 0
            stats.backtracks += 1
        return False

    try:
        ok = rec()
    except _Cap:
        ok = False
    stats.time_s = time.perf_counter() - t0
    return (g if ok else None), stats