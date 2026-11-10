"""Integrated demo.  python run_demo.py [n_misreads] [seed]
Input mosaic -> NN reading -> correction search -> CSP solution, plus ablations."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import config
from src.data.sudoku_gen import make_puzzle
from src.data.mosaic import render_mosaic
from src.data.misreads import make_misread_instance
from src.eval import correction_benchmark as cb
from src.integration.pipeline import solve_from_tiles
from src.nn.interface import predict_cells

k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1


def draw_grid(ax, grid, title, hl=(), color="#f4a6a6"):
    ax.set_xlim(0, 9); ax.set_ylim(9, 0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for (r, c) in hl:
        ax.add_patch(Rectangle((c, r), 1, 1, color=color))
    for i in range(10):
        lw = 2.2 if i % 3 == 0 else 0.6
        ax.plot([0, 9], [i, i], "k", lw=lw)
        ax.plot([i, i], [0, 9], "k", lw=lw)
    for r in range(9):
        for c in range(9):
            if grid[r, c]:
                ax.text(c + .5, r + .5, str(int(grid[r, c])), ha="center", va="center", fontsize=11)
    ax.set_title(title, fontsize=10)


def stat(stats, name):
    return getattr(stats, name, float("nan"))


# 1. input: a generated puzzle rendered as MNIST digits, with k hard digits planted
imgs, pool = cb.load_world()
puzzle, truth = make_puzzle(cb.N_GIVENS, seed)
puzzle = np.asarray(puzzle).reshape(81)
truth = np.asarray(truth).reshape(9, 9)
tiles, injected, idx = make_misread_instance(puzzle, k, seed * 10 + k, pool, imgs)
mosaic = render_mosaic(idx, imgs)

# 2. NN reads the mosaic
probs, blank = predict_cells(tiles)
read = cb.read_grid_from(probs, blank)
print(f"Puzzle: {cb.N_GIVENS} givens, {k} planted misreads (true -> NN guess): {injected}")

# 3. correction search with all three methods (the search trace)
fixed = lambda t, **kw: (probs.copy(), blank.copy())
res = {}
print("\nSearch over digit corrections (NN probabilities as costs):")
for m in ("ucs", "greedy", "astar"):
    res[m] = solve_from_tiles(tiles, method=m, predict=fixed, solver=cb.csp_solve,
                              max_corrections=cb.MAX_CORRECTIONS)
    s = res[m].search
    print(f"  {m.upper():6s} expanded={s.expanded:5d} generated={s.generated:5d} "
          f"cost={s.cost:.3f} corrections={res[m].corrections} status={res[m].status}")
best = res["astar"]
if not best.solved:
    print("\nUNREADABLE:", best.reason, "| least confident cells (row, col, digit, conf):",
          best.suspects)

# 4. CSP ablation on the corrected grid: BT vs BT+FC vs BT+FC+AC-3
grid = best.corrected_grid
ablation = {}
print("\nCSP on the corrected grid:")
for name, kw in [("BT", dict(use_fc=False, use_mrv=False)),
                 ("BT+FC", dict(use_fc=True, use_mrv=False)),
                 ("BT+FC+AC-3", dict(use_fc=True, use_mrv=False, use_ac3=True))]:
    from src.csp.solver import solve
    sol_list, st = solve(grid.flatten().tolist(), max_assignments=200_000, **kw)
    ablation[name] = stat(st, "assignments")
    print(f"  {name:11s} assignments={ablation[name]} backtracks={stat(st, 'backtracks')} "
          f"solved={sol_list is not None}")

# 5. NN on (with correction) vs off (raw reading straight to the CSP)
off = cb.csp_solve(read)
on_ok = best.solved and np.array_equal(best.solution, truth)
print(f"\nNN-guided correction: solved={best.solved}, exact solution recovered={on_ok}")
print(f"No correction       : solved={off is not None}, "
      f"exact solution recovered={off is not None and np.array_equal(off, truth)}")

# 6. figure
fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))
ax[0, 0].imshow(mosaic, cmap="gray"); ax[0, 0].axis("off")
for (r, c) in injected:
    ax[0, 0].add_patch(Rectangle((c * 28, r * 28), 28, 28, fill=False, ec="red", lw=2))
ax[0, 0].set_title("Input mosaic (red = digits the NN misreads)", fontsize=10)
draw_grid(ax[0, 1], read, "NN reading (red = wrong)", hl=set(injected))
draw_grid(ax[0, 2], best.corrected_grid, "After correction search (green = changed)",
          hl={(r, c) for r, c, _, _ in best.corrections}, color="#a6e3a6")
draw_grid(ax[1, 0], best.solution if best.solved else np.zeros((9, 9), int),
          "CSP solution" if best.solved else "UNREADABLE: no solution")
ax[1, 1].bar(["UCS", "Greedy", "A*"], [res[m].search.expanded for m in ("ucs", "greedy", "astar")])
ax[1, 1].set_title("Search: nodes expanded"); ax[1, 1].set_yscale("symlog")
ax[1, 2].bar(list(ablation), [ablation[n] for n in ablation])
ax[1, 2].set_title("CSP: assignments (BT vs FC vs AC-3)"); ax[1, 2].set_yscale("log")
fig.suptitle(f"Demo: NN on -> solved={on_ok}   |   NN off -> "
             f"solved={off is not None and np.array_equal(off, truth)}")
fig.tight_layout()
path = config.ROOT / "results" / "demo.png"
fig.savefig(path, dpi=150)
print("saved", path)
try:
    os.startfile(path)
except Exception:
    pass
