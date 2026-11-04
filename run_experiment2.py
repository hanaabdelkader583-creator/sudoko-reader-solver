"""Run Experiment 2.  From the repo root:  python run_experiment2.py [quick]"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config
from src.eval.benchmark import run_all, CONFIGS

quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
res = run_all(n_per_group=2 if quick else None)

out_dir = config.ROOT / "results"
out_dir.mkdir(exist_ok=True)

lines = ["| Givens | Method | Assignments (mean ± std) | Backtracks (mean ± std) | Time s (mean ± std) | Capped |",
         "|---|---|---|---|---|---|"]
for g, level in res.items():
    for name, m in level.items():
        f = lambda k: f"{m[k].mean():.1f} ± {m[k].std(ddof=1):.1f}"
        lines.append(f"| {g} | {name} | {f('assignments')} | {f('backtracks')} | "
                     f"{m['time_s'].mean():.4f} ± {m['time_s'].std(ddof=1):.4f} | {int(m['capped'].sum())} |")
table = "\n".join(lines)
capped_total = sum(int(res[g][n]["capped"].sum()) for g in res for n in res[g])
n_per_level = config.PUZZLES_PER_LEVEL if not quick else 2 * config.N_SEED_GROUPS
note = (f"\n\nNote: runs are capped at 500,000 assignments. "
        f"{capped_total} puzzle run(s) hit the cap (see the Capped column, "
        f"out of {n_per_level} puzzles per level and method); the means for those "
        f"rows are therefore lower bounds. Std is the sample std across "
        f"{config.N_SEED_GROUPS} seed groups.")
table = table + note
(out_dir / "experiment2_table.md").write_text(table, encoding="utf-8")
print(table)

metrics = [("assignments", "Assignments"),
           ("backtracks", "Backtracks"),
           ("time_s", "Time (s)")]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
width = 0.25
for ax, (key, label) in zip(axes, metrics):
    for k, name in enumerate(CONFIGS):
        means = [res[g][name][key].mean() for g in config.GIVENS_LEVELS]
        stds = [res[g][name][key].std(ddof=1) for g in config.GIVENS_LEVELS]
        ax.bar(np.arange(3) + k * width, means, width,
               yerr=stds, capsize=3, label=name)
    ax.set_xticks(np.arange(3) + width)
    ax.set_xticklabels([str(g) for g in config.GIVENS_LEVELS])
    ax.set_xlabel("Number of givens")
    ax.set_ylabel(f"{label} (mean over 5 seeds)")
    ax.set_yscale("log")
    ax.set_title(label)
axes[0].legend()
fig.suptitle("Experiment 2: BT vs BT+FC vs BT+FC+AC-3")
fig.tight_layout()
fig.savefig(out_dir / "experiment2_metrics.png", dpi=150)
print("saved to", out_dir)