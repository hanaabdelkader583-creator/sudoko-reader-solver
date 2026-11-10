"""Experiment 1: UCS vs Greedy vs A*.  python run_experiment1.py [quick]"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config
from src.eval import correction_benchmark as cb

quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
n_per_group = 2 if quick else config.PUZZLES_PER_GROUP
METHODS = ("ucs", "greedy", "astar")

records, skipped = [], 0
for inst in cb.iter_instances(n_per_group):
    if inst["skipped"]:
        skipped += 1
        continue
    for m in METHODS:
        records.append(cb.run_search(inst, m))
    print(f"\rgroup {inst['group']} puzzle {inst['puzzle']} misreads {inst['level']}   ",
          end="", flush=True)
print()

out = config.ROOT / "results"
out.mkdir(exist_ok=True)
rec, sol = cb.aggregate(records, "recovered"), cb.aggregate(records, "solved")
exp, gen = cb.aggregate(records, "expanded"), cb.aggregate(records, "generated")
nc, tm = cb.aggregate(records, "n_corr"), cb.aggregate(records, "time_s")

lines = ["| Misreads | Method | Recovery % | Solved % | Nodes expanded | Nodes generated | Corrections | Time s |",
         "|---|---|---|---|---|---|---|---|"]
for k in cb.MISREAD_LEVELS:
    for m in METHODS:
        if (k, m) in rec:
            lines.append(f"| {k} | {m.upper()} | {cb.pm(rec[k, m], 100)} | {cb.pm(sol[k, m], 100)} | "
                         f"{cb.pm(exp[k, m])} | {cb.pm(gen[k, m])} | {cb.pm(nc[k, m], 1, 2)} | "
                         f"{cb.pm(tm[k, m], 1, 4)} |")
n_inst = len(records) // len(METHODS)
note = (f"\n\nNote: {cb.N_GIVENS} givens, {n_inst} instances evaluated "
        f"({skipped} skipped: not enough hard images for that puzzle), "
        f"{config.N_SEED_GROUPS} seed groups; std is the sample std across groups. "
        f"Corrections are capped at {cb.MAX_CORRECTIONS} (project brief), so 4 misreads "
        f"cannot be fully repaired. Recovery = the final solution equals the true solution.")
table = "\n".join(lines) + note
(out / "experiment1_table.md").write_text(table, encoding="utf-8")
cb.write_log(records, out / "correction_log.csv")
print(table)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, (data, label, scale) in zip(axes, [(rec, "Recovery rate (%)", 100),
                                            (exp, "Nodes expanded", 1),
                                            (tm, "Time (s)", 1)]):
    for m in METHODS:
        ks = [k for k in cb.MISREAD_LEVELS if (k, m) in data]
        ax.errorbar(ks, [data[k, m][0] * scale for k in ks],
                    yerr=[data[k, m][1] * scale for k in ks], marker="o", capsize=3, label=m.upper())
    ax.set_xlabel("Injected misreads")
    ax.set_ylabel(label)
    ax.set_title(label)
axes[1].set_yscale("symlog")
axes[2].set_yscale("symlog", linthresh=1e-3)
axes[0].legend()
fig.suptitle("Experiment 1: UCS vs Greedy vs A* over digit corrections")
fig.tight_layout()
fig.savefig(out / "experiment1_metrics.png", dpi=150)
print("saved to", out)
