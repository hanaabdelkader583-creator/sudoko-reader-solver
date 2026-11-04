"""Experiment 2: BT vs BT+FC vs BT+FC+AC-3 on 25/30/35 givens."""
import numpy as np
import config
from src.csp.solver import solve

CONFIGS = {
    "BT":          dict(use_fc=False, use_mrv=False),
    "BT+FC":       dict(use_fc=True,  use_mrv=False),
    "BT+FC+AC-3":  dict(use_fc=True,  use_mrv=False, use_ac3=True),
}
CAP = 500_000


def run_level(givens, n_per_group=None):
    """Return {config_name: {'assignments': (5,), 'backtracks': (5,), 'time_s': (5,), 'capped': (5,)}}.
    Each entry is the per-seed-group mean over that group's puzzles."""
    d = np.load(config.SUDOKU_DIR / f"puzzles_g{givens}.npz")
    P, S, grp = d["puzzles"], d["solutions"], d["seed_group"]
    out = {}
    for name, kw in CONFIGS.items():
        rows = {"assignments": [], "backtracks": [], "time_s": [], "capped": []}
        for g in range(config.N_SEED_GROUPS):
            idx = np.flatnonzero(grp == g)[:n_per_group]
            a, b, t, c = [], [], [], 0
            for i in idx:
                sol, st = solve(P[i], max_assignments=CAP, **kw)
                if sol is None:                       # cap hit (puzzles are solvable)
                    c += 1
                else:
                    assert sol == [int(v) for v in S[i]]
                a.append(st.assignments); b.append(st.backtracks); t.append(st.time_s)
            rows["assignments"].append(np.mean(a))
            rows["backtracks"].append(np.mean(b))
            rows["time_s"].append(np.mean(t))
            rows["capped"].append(c)
        out[name] = {k: np.array(v) for k, v in rows.items()}
    return out


def run_all(n_per_group=None):
    return {g: run_level(g, n_per_group) for g in config.GIVENS_LEVELS}