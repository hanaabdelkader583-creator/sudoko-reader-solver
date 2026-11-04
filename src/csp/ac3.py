"""AC-3 arc consistency for Sudoku. Domains are a list of 81 sets, edited in place."""
from collections import deque
from src.csp.model import PEERS


def ac3(dom, stats, queue=None, trail=None):
    """Make domains arc consistent.
    queue: cells whose domain just changed (None = all cells).
    trail: optional list; every removal is appended as (cell, value) so the
           caller can undo it on backtrack.
    Returns False if some domain becomes empty (no solution), else True."""
    q = deque(range(81) if queue is None else queue)
    while q:
        y = q.popleft()
        if len(dom[y]) != 1:
            continue                      # only a single-value cell can remove from peers
        (v,) = dom[y]
        for x in PEERS[y]:
            stats.ac3_revisions += 1
            if v in dom[x]:
                dom[x].discard(v)
                stats.prunings += 1
                if trail is not None:
                    trail.append((x, v))
                if not dom[x]:
                    return False
                q.append(x)               # x changed, so re-check its peers
    return True