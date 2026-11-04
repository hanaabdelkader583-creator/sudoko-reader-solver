"""Switchable CSP solver: BT, BT+FC, +MRV, +AC-3, optionally maintained."""
import time
from src.csp.ac3 import ac3
from src.csp.model import PEERS, to_list, find_conflicts, initial_domains
from src.csp.stats import Stats


class _Cap(Exception):
    pass


def solve(grid, use_fc=True, use_mrv=True, use_degree=False, use_ac3=False,
          maintain_ac3=False, max_assignments=None):
    """Return (solution_list or None, Stats)."""
    if (use_ac3 or maintain_ac3) and not use_fc:
        raise ValueError("use_ac3 needs use_fc (domains must be tracked)")
    if use_degree and not use_mrv:
        raise ValueError("use_degree is a tie-break for MRV, so it needs use_mrv")

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

    if use_ac3:
        if not ac3(dom, stats):
            stats.time_s = time.perf_counter() - t0
            return None, stats

    def pick():
        blanks = [i for i in range(81) if g[i] == 0]
        if not blanks:
            return None
        if use_mrv:
            if use_degree:
                return min(blanks, key=lambda i: (
                    len(dom[i]),
                    -sum(1 for j in PEERS[i] if g[j] == 0)))
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
            trail = []
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
            if ok and maintain_ac3:
                saved = dom[i]
                dom[i] = {v}
                ok = ac3(dom, stats, queue=[i], trail=trail)
                dom[i] = saved
            if ok and rec():
                return True
            for x, val in trail:               # undo AC-3 pruning
                dom[x].add(val)
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