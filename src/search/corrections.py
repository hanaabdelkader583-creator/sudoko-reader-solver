"""
src/search/corrections.py
=========================
Search over digit corrections: UCS, Greedy and A*.

Input : NN output (probs (81, 9) over digits 1-9, blank (81,) bool).
Output: the cheapest set of digit corrections that makes the read grid
        free of row/column/box conflicts.

State      = set of corrections {(cell, new_digit)}; each cell is corrected at most once.
Step cost  = log p(read digit) - log p(new digit)   (>= 0, since read = argmax)
Goal       = no duplicate digit in any row, column or box.
Branching  = take the first conflicting pair (i, j); any solution must change i or j,
             so branching on those two cells is complete.
Heuristic  = sum over a set of cell-disjoint conflicting pairs of the cheapest
             correction of either cell. Admissible, so A* is optimal.
"""
from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field

import numpy as np

EPS = 1e-6
METHODS = ("ucs", "greedy", "astar")


def _build_pairs() -> list[tuple[int, int]]:
    pairs = set()
    for r in range(9):
        for c in range(9):
            i = r * 9 + c
            for k in range(81):
                if k <= i:
                    continue
                r2, c2 = divmod(k, 9)
                if r == r2 or c == c2 or (r // 3 == r2 // 3 and c // 3 == c2 // 3):
                    pairs.add((i, k))
    return sorted(pairs)


PAIRS = _build_pairs()


def find_conflicts(grid) -> list[tuple[int, int]]:
    """Pairs of flat cell indices (i, j) holding the same non-zero digit in one unit."""
    g = np.asarray(grid).reshape(81)
    return [(i, j) for i, j in PAIRS if g[i] != 0 and g[i] == g[j]]


@dataclass
class SearchResult:
    found: bool
    grid: np.ndarray                      # 9x9, 0 = blank
    corrections: list = field(default_factory=list)   # (row, col, read, new)
    cost: float = 0.0
    expanded: int = 0
    generated: int = 0
    method: str = "astar"


def _cost_table(probs: np.ndarray, base: np.ndarray, blank: np.ndarray) -> np.ndarray:
    logp = np.log(np.maximum(probs, EPS))
    cost = np.full((81, 9), math.inf)
    for i in range(81):
        if blank[i]:
            continue
        cost[i] = np.maximum(0.0, logp[i, base[i] - 1] - logp[i])
        cost[i, base[i] - 1] = math.inf     # "keeping" the digit is not a correction
    return cost


def _heuristic(pairs, cost, corrected_cells) -> float:
    used, h = set(), 0.0
    for i, j in pairs:
        if i in used or j in used:
            continue
        ci = math.inf if i in corrected_cells else cost[i].min()
        cj = math.inf if j in corrected_cells else cost[j].min()
        m = min(ci, cj)
        if m == math.inf:
            return math.inf                 # dead end: neither cell can change
        h += m
        used.update((i, j))
    return h


def search_corrections(probs, blank, method: str = "astar",
                       max_expansions: int = 200_000, top_k: int = 8,
                       max_corrections: int | None = None, goal=None,
                       n_uncertain: int = 8) -> SearchResult:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    probs = np.asarray(probs, dtype=np.float64).reshape(81, 9)
    blank = np.asarray(blank, dtype=bool).reshape(81)
    base = np.where(blank, 0, probs.argmax(axis=1) + 1)
    cost = _cost_table(probs, base, blank)
    cands = [list(np.argsort(cost[i])[:top_k]) if not blank[i] else [] for i in range(81)]

    def apply(state):
        g = base.copy()
        for cell, d in state:
            g[cell] = d
        return g

    def priority(g_cost, h):
        return {"ucs": g_cost, "greedy": h, "astar": g_cost + h}[method]

    uncertain = sorted((i for i in range(81) if cands[i]),
                       key=lambda i: cost[i, cands[i][0]])[:n_uncertain]
    start = frozenset()
    pairs0 = find_conflicts(base)
    h0 = _heuristic(pairs0, cost, set())
    counter = itertools.count()
    heap = [(priority(0.0, h0), next(counter), 0.0, start)]
    best_g = {start: 0.0}
    expanded, generated = 0, 1

    while heap:
        _, _, g_cost, state = heapq.heappop(heap)
        if g_cost > best_g.get(state, math.inf) + 1e-12:
            continue
        grid = apply(state)
        pairs = find_conflicts(grid)
        if not pairs and (goal is None or goal(grid.reshape(9, 9))):
            corr = sorted((c // 9, c % 9, int(base[c]), d) for c, d in state)
            return SearchResult(True, grid.reshape(9, 9), corr, g_cost, expanded, generated, method)
        if expanded >= max_expansions:
            break
        if max_corrections is not None and len(state) >= max_corrections:
            continue
        expanded += 1
        corrected = {c for c, _ in state}
        for cell in (pairs[0] if pairs else uncertain):
            if cell in corrected:
                continue
            for k in cands[cell]:
                step = cost[cell, k]
                if step == math.inf:
                    continue
                ns = state | {(cell, int(k) + 1)}
                ng = g_cost + step
                if ng >= best_g.get(ns, math.inf):
                    continue
                h = _heuristic(find_conflicts(apply(ns)), cost, corrected | {cell})
                if h == math.inf:
                    continue
                best_g[ns] = ng
                generated += 1
                heapq.heappush(heap, (priority(ng, h), next(counter), ng, ns))

    return SearchResult(False, base.reshape(9, 9), [], math.inf, expanded, generated, method)