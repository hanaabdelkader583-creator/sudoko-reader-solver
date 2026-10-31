"""CSP model for 9x9 Sudoku. Cells are indexed 0-80 (row-major), 0 = blank."""

ROW = [i // 9 for i in range(81)]
COL = [i % 9 for i in range(81)]
BOX = [(i // 27) * 3 + (i % 9) // 3 for i in range(81)]

# The 27 all-different groups.
UNITS = (
    [[r * 9 + c for c in range(9)] for r in range(9)]                      # rows
    + [[r * 9 + c for r in range(9)] for c in range(9)]                    # columns
    + [[i for i in range(81) if BOX[i] == b] for b in range(9)]            # boxes
)

# PEERS[i] = the 20 cells that must differ from cell i.
PEERS = [
    [j for j in range(81)
     if j != i and (ROW[j] == ROW[i] or COL[j] == COL[i] or BOX[j] == BOX[i])]
    for i in range(81)
]


def to_list(grid):
    """Accept a numpy array or list; return a plain list of 81 ints (0-9)."""
    g = [int(v) for v in grid]
    if len(g) != 81 or any(v < 0 or v > 9 for v in g):
        raise ValueError("grid must have 81 values in 0-9")
    return g


def find_conflicts(grid):
    """Return (i, j) pairs, i < j, of peer cells holding the same nonzero digit."""
    g = to_list(grid)
    return [(i, j) for i in range(81) if g[i]
            for j in PEERS[i] if j > i and g[j] == g[i]]


def is_solved(grid):
    """True if every cell is filled and there are no conflicts."""
    g = to_list(grid)
    return all(g) and not find_conflicts(g)


def initial_domains(grid):
    """One set per cell: {v} for a given, {1..9} for a blank.
    No propagation here; pruning is the job of FC and AC-3."""
    g = to_list(grid)
    return [{v} if v else set(range(1, 10)) for v in g]