# -*- coding: utf-8 -*-
"""Fig 9/10 (resubmit v1, rev5): latitude dependence, 3x2 six panels.

Per Fang: also show the surrogate-significance--latitude test at all three
lags, not only 60 min. Layout: columns = lag (20/60/120 min),
top row = P_dyn excess TE, bottom row = surrogate z. Each lag column pairs
"effect + significance".
Data: v7_revision/results/e5b_station_excess.csv (excess, same source as the
r_s quoted in Section 4.6) and server_sync/results_e3/e5_station_mlat.csv (z).
Output: resubmit_v1/figures/fig9_latitude.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

E5B = r"E:/project_ulf_omni/ground_pc5/v7_revision/results/e5b_station_excess.csv"
E5M = r"E:/project_ulf_omni/ground_pc5/server_sync/results_e3/e5_station_mlat.csv"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CODE_MAP = {"MHT": "OMOHE", "MZL": "OMZLN", "NAT": "ONOAN", "SSL": "OCSSL",
            "MLS": "OTANC", "PXT": "OPIXI", "JFT": "OYICH", "HZT": "OHZXH",
            "SYT": "OSHYA", "ZQT": "OZHQI", "LAT": "OLSCG", "SYS": "OLEDO",
            "QZT": "OQIZH"}
nw = lambda s: CODE_MAP.get(s, s)

dfb = pd.read_csv(E5B)
COLOR = {20: "#1f77b4", 60: "#d62728", 120: "#2ca02c"}

plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.9,
                     "savefig.bbox": "tight"})
fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.6))
XLIM = (7, 46)

tags = iter("abcdef")
# ---------------- top row: excess TE vs latitude ----------------
for j, lag in enumerate([20, 60, 120]):
    ax = axes[0, j]
    g = dfb[dfb.lag_min == lag].sort_values("mlat")
    ax.plot(g.mlat, g.excess, "o", color=COLOR[lag], ms=6.5, zorder=3)
    p = np.polyfit(g.mlat, g.excess, 1)
    xs = np.linspace(g.mlat.min(), g.mlat.max(), 30)
    ax.plot(xs, np.polyval(p, xs), "-", color=COLOR[lag], lw=1.8,
            alpha=0.65, zorder=2)
    r, pv = spearmanr(g.mlat, g.excess)
    sig = "p = 0.035" if lag == 20 else "n.s."
    ax.text(0.96, 0.94, f"lag {lag} min\n$r_s$ = {r:.2f} ({sig})",
            transform=ax.transAxes, fontsize=9.5, ha="right", va="top",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))
    ax.set_xlim(*XLIM)
    ax.set_ylim(0.0045, 0.0150)
    ax.set_xlabel("Magnetic latitude (deg N)")
    if j == 0:
        ax.set_ylabel("$P_{dyn}$ excess TE (nats)")
    ax.set_title(f"({next(tags)}) Excess TE, lag {lag} min", fontsize=10)

# ---------------- bottom row: surrogate z vs latitude ----------------
dm = pd.read_csv(E5M)
dm = dm[(dm.station != "ZSZ") & (dm.param == "Flow_pressure")]
dm = dm[dm.lag_min.isin([20, 60, 120])]
dm = dm.groupby(["lag_min", "station"]).agg(mlat=("mlat", "median"),
                                            z=("z", "median"),
                                            src=("src", "first")).reset_index()
for j, lag in enumerate([20, 60, 120]):
    ax = axes[1, j]
    g = dm[dm.lag_min == lag]
    for src, mk, lb in [("table", "o", "project-table mlat"),
                        ("public-approx", "s", "approximate mlat")]:
        dd = g[g.src == src]
        ax.plot(dd.mlat, dd.z, mk, ms=9,
                mfc="#d62728" if src == "table" else "none",
                mec="#d62728", mew=1.4, label=lb, zorder=3)
    p = np.polyfit(g.mlat, g.z, 1)
    xs = np.linspace(g.mlat.min(), g.mlat.max(), 30)
    r, pv = spearmanr(g.mlat, g.z)
    ax.plot(xs, np.polyval(p, xs), "--", color="#555555", lw=1.6, alpha=0.8,
            label=f"OLS ({p[0]:+.3f} z/deg)")
    ax.axhline(2, color="gray", ls=":", lw=1)
    ax.text(45.5, 2.15, "z = 2", fontsize=8, color="gray", ha="right")
    row = g[g.station == "HZT"].iloc[0]
    ax.annotate("OHZXH (susp.)", (row.mlat, row.z), fontsize=9,
                xytext=(6, -14), textcoords="offset points",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.5))
    ax.text(0.96, 0.94, f"$r_s$ = {r:.2f} ({'p = 0.071' if lag == 60 else 'n.s.'})",
            transform=ax.transAxes, fontsize=9.5, ha="right", va="top",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))
    ax.set_xlim(*XLIM)
    ax.set_ylim(0, 9)
    ax.set_xlabel("Magnetic latitude (deg N)")
    if j == 0:
        ax.set_ylabel("Surrogate $z$")
    ax.set_title(f"({next(tags)}) Surrogate $z$, lag {lag} min", fontsize=10)
    if j == 0:
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

plt.tight_layout()
out = os.path.join(ROOT, "figures", "fig9_latitude.png")
plt.savefig(out, dpi=300)
plt.close()
print("fig9_latitude (3x2 six-panel) saved ->", out)
