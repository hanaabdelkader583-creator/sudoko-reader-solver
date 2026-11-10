"""Integrated demo:  python run_demo.py            (real MNIST + trained CNN)
                    python run_demo.py --synthetic (no data needed, no mosaic picture)

Shows: input (read grid) -> AC-3 conflict -> search trace (UCS/Greedy/A*) -> corrections ->
solution, then plots how NN + Search + CSP cooperate, CSP pruning with/without FC/AC-3, and
NN on versus off.  Figure saved to outputs/demo/demo.png
"""
import argparse, os, random
import numpy as np
import matplotlib
from sudoku_ai.csp import solve, grid_to_str, cell_name
from sudoku_ai.generator import make_puzzle, build_pools, sample_instance
from sudoku_ai.integration import correct_and_solve, solve_no_correction, solve_random_correction
from sudoku_ai.synthetic import synthetic_probs


def draw_grid(ax, grid, title, mark=None, mark2=None):
    mark, mark2 = mark or set(), mark2 or set()
    ax.set_xlim(0, 9); ax.set_ylim(9, 0); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for k in range(10):
        lw = 2.2 if k % 3 == 0 else 0.5
        ax.plot([0, 9], [k, k], "k", lw=lw); ax.plot([k, k], [0, 9], "k", lw=lw)
    for i, v in enumerate(grid):
        if v:
            col = "red" if i in mark else ("tab:blue" if i in mark2 else "black")
            ax.text(i % 9 + .5, i // 9 + .5, str(v), ha="center", va="center", fontsize=13, color=col,
                    fontweight="bold" if i in mark else "normal")
    ax.set_title(title, fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--misreads", type=int, default=2)
    ap.add_argument("--givens", type=int, default=30)
    ap.add_argument("--unique", action="store_true", help="extension: require exactly one solution")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()
    if a.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X = None
    if a.synthetic:
        probs, y = synthetic_probs()
    else:
        from sudoku_ai.data import load_dataset, to_float, build_mosaic, split_mosaic, is_blank_tile
        probs, y = np.load("artifacts/test_probs.npz")["cnn_cal"], np.load("artifacts/test_probs.npz")["y"]
        X = to_float(load_dataset()["X_test"])
    pools = build_pools(probs, y)
    rng = random.Random(a.seed)
    puzzle, truth = make_puzzle(rng, a.givens)
    inst = sample_instance(puzzle, probs, pools, a.misreads, rng)
    blank = inst["blank"]
    mosaic = None
    if X is not None:                                  # blank cells found by the brightness rule
        mosaic = build_mosaic(inst["idx"], X)
        blank = [is_blank_tile(t) for t in split_mosaic(mosaic)]
        assert blank == inst["blank"]
    mis = set(inst["misread_cells"])

    print("=== INPUT: grid as read by the network (0/'.' = blank) ===")
    r = correct_and_solve(inst["probs81"], blank, algo="ucs", record_trace=True, uniqueness=a.unique)
    print(grid_to_str(r.read_grid))
    print("\n[AC-3 on first reading]", r.ac3_reason)
    print("[Injected misreads (hidden from solver)]", [cell_name(c) for c in sorted(mis)])
    print("\n=== SEARCH TRACE (UCS expansion order: corrections, cost g) ===")
    for n, st, g, h in r.trace[:15]:
        print(f"  #{n:<3d} g={g:6.3f}  corrections={[(cell_name(c), d) for c, d in st]}")
    print("\n=== RESULT ===\n", r.message)
    for c in r.corrections:
        print(f"  {c['cell']}: read {c['read']} -> {c['new']}  (cost {c['cost']:.3f})")
    if r.solution:
        print(grid_to_str(r.solution)); print("Exact match with true solution:", r.solution == truth)

    algos = {k: correct_and_solve(inst["probs81"], blank, algo=k, record_trace=True) for k in ("ucs", "greedy", "astar")}
    corrected = list(r.read_grid)
    for c in r.corrections:
        corrected[c["index"]] = c["new"]
    cfg = {"BT": {}, "BT+FC": {"fc": True}, "BT+AC3": {"ac3": True}, "BT+FC+AC3": {"fc": True, "ac3": True}}
    csp_cnt = {k: solve(corrected, **kw).counters for k, kw in cfg.items()}
    print("\n=== CSP pruning on the corrected puzzle ===")
    for k, c in csp_cnt.items():
        print(f"  {k:10s} assignments={c.assignments:6d} backtracks={c.backtracks:6d} prunes={c.domain_prunes}")
    mini = {"NN-guided (UCS)": 0, "no correction": 0, "random correction": 0}
    n_mini = 15
    for _ in range(n_mini):
        p2, t2 = make_puzzle(rng, a.givens)
        i2 = sample_instance(p2, probs, pools, a.misreads, rng)
        mini["NN-guided (UCS)"] += correct_and_solve(i2["probs81"], i2["blank"]).solution == t2
        mini["no correction"] += solve_no_correction(i2["probs81"], i2["blank"]) == t2
        mini["random correction"] += solve_random_correction(i2["probs81"], i2["blank"], rng) == t2
    print(f"\n=== NN on vs off ({n_mini} puzzles, {a.misreads} misreads): exact recoveries ===")
    for k, v in mini.items():
        print(f"  {k:18s} {v}/{n_mini}")

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(2, 4, 1)
    if mosaic is not None:
        ax.imshow(mosaic, cmap="gray")
        for c in mis:
            ax.add_patch(plt.Rectangle((c % 9 * 28, c // 9 * 28), 28, 28, fill=False, ec="red", lw=2))
        ax.set_title("Handwritten mosaic (red = injected hard digit)", fontsize=9)
    else:
        ax.text(.5, .5, "mosaic needs real MNIST\n(--synthetic mode)", ha="center"); ax.set_title("Mosaic", fontsize=9)
    ax.axis("off")
    draw_grid(fig.add_subplot(2, 4, 2), r.read_grid, "NN first reading (red = truly misread)", mark=mis)
    draw_grid(fig.add_subplot(2, 4, 3), corrected, "After search corrections (blue = changed)", mark2={c["index"] for c in r.corrections})
    draw_grid(fig.add_subplot(2, 4, 4), r.solution or [0] * 81, "CSP solution" if r.solution else "UNREADABLE")
    ax = fig.add_subplot(2, 4, 5)
    for k, res in algos.items():
        ax.plot([t[0] for t in res.trace], [t[2] for t in res.trace], marker=".", label=f"{k} ({res.search['expanded']} exp.)")
    ax.set(xlabel="expansion number", ylabel="path cost g", title="Search trace"); ax.legend(fontsize=7)
    ax = fig.add_subplot(2, 4, 6)
    names = list(algos); ax.bar(names, [algos[k].search["generated"] for k in names], color="tab:orange")
    ax.set_title("Nodes generated (corrections explored)", fontsize=9)
    ax = fig.add_subplot(2, 4, 7)
    ax.bar(list(csp_cnt), [c.backtracks + 1 for c in csp_cnt.values()], color="tab:green")
    ax.set_yscale("log"); ax.set_title("CSP backtracks (+1, log): with/without FC, AC-3", fontsize=9); ax.tick_params(axis="x", labelsize=7)
    ax = fig.add_subplot(2, 4, 8)
    ax.bar(list(mini), [v / n_mini for v in mini.values()], color=["tab:blue", "tab:gray", "tab:red"])
    ax.set_ylim(0, 1); ax.set_title(f"NN on vs off: exact recovery ({a.misreads} misreads)", fontsize=9); ax.tick_params(axis="x", labelsize=7)
    plt.tight_layout()
    os.makedirs("outputs/demo", exist_ok=True)
    plt.savefig("outputs/demo/demo.png", dpi=130)
    print("\nfigure saved to outputs/demo/demo.png")
    if not a.no_show:
        plt.show()


if __name__ == "__main__":
    main()