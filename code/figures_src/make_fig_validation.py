# -*- coding: utf-8 -*-
"""Resubmit v1: regenerate fig2_validation.png (-> Figure 3) fixing the
y-label overlap in panels (b) and (c) reported by the editor round:
y-labels were flush against the axes ("Null percentile (nats)" / "nats").
Fix: constrained layout + wider figure + explicit labelpad.

Values are identical to the server-generated original
(server_sync/make_paper_figures.py, Fig 2 section; benchmark & null
percentiles from E1, results in server_sync/results_e2/e1_validation.csv).
Output: resubmit_v1/figures/fig2_validation.png (overwrite in place)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9})
OUT = r"E:\project_ulf_omni\ground_pc5\resubmit_v1\figures"
os.makedirs(OUT, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True)

# (a) coupled-AR benchmark (E1a): TE(x->y)=0.0585, reverse -0.0003
ax = axes[0]
ax.bar([0, 1], [0.0585, -0.0003], color=["#d62728", "#7f7f7f"], width=0.5)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks([0, 1]); ax.set_xticklabels(["TE(X$\\to$Y)", "TE(Y$\\to$X)"])
ax.set_ylabel("TE (nats)", labelpad=6)
ax.set_title("(a) Coupled-AR benchmark (JIDT)", fontsize=9)

# (b) null calibration percentiles (E1b, n=2000)
ax = axes[1]
nulls = {"iid": [0.0224, 0.0265], "shift (dest)": [0.0179, 0.0241]}
x = np_range = [0, 1]
w = 0.35
for i, (k, v) in enumerate(nulls.items()):
    ax.bar([xi + (i - 0.5) * w for xi in x], v, w, label=k,
           color=["#1f77b4", "#ff7f0e"][i])
ax.set_xticks(x); ax.set_xticklabels(["p95", "p99"])
ax.set_ylabel("Null percentile (nats)", labelpad=6)
ax.set_title("(b) Null calibration (n=2000)", fontsize=9)
ax.legend(fontsize=8)

# (c) scale contrast: typical Pdyn excess vs null p95
ax = axes[2]
ax.bar([0, 1], [0.0224, 0.0179], color=["#1f77b4", "#ff7f0e"], width=0.5)
ax.axhline(0.0060, color="#d62728", lw=1.5)
ax.text(0.5, 0.0090, "typical P$_{dyn}$ excess\n(0.006 nats)", color="#d62728",
        fontsize=7, ha="center")
ax.set_xticks([0, 1]); ax.set_xticklabels(["iid null p95", "shift null p95"])
ax.set_ylabel("nats", labelpad=6)
ax.set_title("(c) Scale contrast: excess vs null", fontsize=9)

plt.savefig(os.path.join(OUT, "fig2_validation.png"), dpi=150)
plt.close()
print("fig2_validation regenerated (y-label overlap fixed)")
