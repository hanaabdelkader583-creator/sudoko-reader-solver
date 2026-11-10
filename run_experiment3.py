"""Experiment 3: NN-guided correction vs no correction vs random correction.
python run_experiment3.py [quick]"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config
from src.eval import correction_benchmark as cb

quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
n_per_group = 2 if quick else config.PUZZLES_PER_GROUP
LABELS = {"astar": "NN-guided (A*)", "none": "No correction", "random": "Random correction"}

records, skipped = [], 0
for inst in cb.iter_instances(n_per_group):
    if inst["skipped"]:
        skipped += 1
        continue
    a = cb.run_search(inst, "astar")
    for rec in (a, cb.run_none(inst), cb.run_random(inst)):
        rec["method"] = LABELS[rec["method"]]
        records.append(rec)
    print(f"\rgroup {inst['group']} puzzle {inst['puzzle']} misreads {inst['level']}   ",
          end="", flush=True)
print()

out = config.ROOT / "results"
out.mkdir(exist_ok=True)
rec, sol, tm = (cb.aggregate(records, k) for k in ("recovered", "solved", "time_s"))

lines = ["| Misreads | Method | Recovery % | Solved % | Time s |", "|---|---|---|---|---|"]
for k in cb.MISREAD_LEVELS:
    for m in LABELS.values():
        if (k, m) in rec:
            lines.append(f"| {k} | {m} | {cb.pm(rec[k, m], 100)} | {cb.pm(sol[k, m], 100)} | "
                         f"{cb.pm(tm[k, m], 1, 4)} |")
note = (f"\n\nNote: same instances as Experiment 1 ({cb.N_GIVENS} givens, {skipped} skipped, "
        f"{config.N_SEED_GROUPS} seed groups, std across groups). Random correction: up to "
        f"{cb.RANDOM_TRIES} attempts of 1-{cb.MAX_CORRECTIONS} random changes to conflicting "
        f"cells; the first conflict-free grid goes to the CSP. 'Solved' = some valid solution; "
        f"'Recovery' = the true solution.")
table = "\n".join(lines) + note
(out / "experiment3_table.md").write_text(table, encoding="utf-8")
print(table)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (data, label) in zip(axes, [(rec, "Recovery rate (%)"), (sol, "Solved rate (%)")]):
    for m in LABELS.values():
        ks = [k for k in cb.MISREAD_LEVELS if (k, m) in data]
        ax.errorbar(ks, [data[k, m][0] * 100 for k in ks], yerr=[data[k, m][1] * 100 for k in ks],
                    marker="o", capsize=3, label=m)
    ax.set_xlabel("Injected misreads")
    ax.set_ylabel(label)
    ax.set_title(label)
axes[0].legend()
fig.suptitle("Experiment 3: NN-guided vs no vs random correction")
fig.tight_layout()
fig.savefig(out / "experiment3_metrics.png", dpi=150)
print("saved to", out)
