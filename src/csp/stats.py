"""Counters shared by all CSP solvers."""
from dataclasses import dataclass, asdict


@dataclass
class Stats:
    assignments: int = 0     # times a value was tried for a cell
    backtracks: int = 0      # times a branch failed and was undone
    prunings: int = 0        # values removed from domains by FC / AC-3
    ac3_revisions: int = 0   # arc revisions done by AC-3
    time_s: float = 0.0      # wall-clock seconds (set by the caller)

    def as_dict(self):
        return asdict(self)