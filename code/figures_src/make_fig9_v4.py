# -*- coding: utf-8 -*-
"""Fig 9/10 (resubmit v1, rev4): latitude dependence, one panel per lag.

Per Fang: split the three lags into separate panels so each trend is read
directly. 2x2 layout:
  (a) lag 20 min  -- the text's primary lag (Spearman r_s=-0.59, p=0.035)
  (b) lag 60 min  -- no significant latitude gradient (r_s=-0.13, n.s.)
  (c) lag 120 min -- no significant latitude gradient (r_s=-0.13, n.s.)
  (d) surrogate z at lag 60 min (table vs approximate mlat) + OLS fit
Data: v7_revision/results/e5b_station_excess.csv (same source as the r_s
quoted in Section 4.6) and server_sync/results_e3/e5_station_mlat.csv (z).
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

df = pd.read_csv(E5B)
COLOR = {20: "#1f77b4", 60: "#d62728", 120: "#2ca02c"}

plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.9,
                     "savefig.bbox": "tight"})
fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.6))
XLIM, YLIM = (7, 46), (0.0045, 0.0150)

PANELS = [("a", 20), ("b", 60), ("c", 120)]
for ax, (tag, lag) in zip(axes.flat[:3], PANELS):
    g = df[df.lag_min == lag].sort_values("mlat")
    ax.plot(g.mlat, g.excess, "o", color=COLOR[lag], ms=6.5, zorder=3)
    p = np.polyfit(g.mlat, g.excess, 1)
    xs = np.linspace(g.mlat.min(), g.mlat.max(), 30)
    ax.plot(xs, np.polyval(p, xs), "-", color=COLOR[lag], lw=1.8, alpha=0.65, zorder=2)
    r, pv = spearmanr(g.mlat, g.excess)
    sig = "p = 0.035" if lag == 20 else "n.s."
    ax.text(0.97, 0.95, f"lag {lag} min\n$r_s$ = {r:.2f} ({sig})",
            transform=ax.transAxes, fontsize=9.5, ha="right", va="top",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_xlabel("Magnetic latitude (deg N)")
    ax.set_ylabel("$P_{dyn}$ excess TE (nats)")
    ax.set_title(f"({tag}) Lag {lag} min", fontsize=10)

# ---------------- (d) surrogate z vs latitude, lag 60 ----------------
ax = axes.flat[3]
dm = pd.read_csv(E5M)
dm = dm[(dm.station != "ZSZ") & (dm.param == "Flow_pressure") & (dm.lag_min == 60)]
d = dm.groupby("station").agg(mlat=("mlat", "median"), z=("z", "median"),
                              src=("src", "first"))
for src, mk, lb in [("table", "o", "project-table mlat"),
                    ("public-approx", "s", "approximate mlat")]:
    dd = d[d.src == src]
    ax.plot(dd.mlat, dd.z, mk, ms=9, mfc="#d62728" if src == "table" else "none",
            mec="#d62728", mew=1.4, label=lb, zorder=3)
p = np.polyfit(d.mlat, d.z, 1)
xs = np.linspace(d.mlat.min(), d.mlat.max(), 30)
ax.plot(xs, np.polyval(p, xs), "--", color="#555555", lw=1.6, alpha=0.8,
        label=f"OLS fit ({p[0]:+.3f} z/deg)")
ax.axhline(2, color="gray", ls=":", lw=1)
ax.text(d.mlat.max() + 0.3, 2.1, "z = 2", fontsize=8, color="gray")
g = d.loc["HZT"]
ax.annotate("OHZXH (susp.)", (g.mlat, g.z), fontsize=9, xytext=(6, -16),
            textcoords="offset points",
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.5))
ax.set_xlabel("Magnetic latitude (deg N)")
ax.set_ylabel("Surrogate $z$ (lag 60 min)")
ax.set_ylim(0, 8.4)
ax.set_title("(d) Significance vs latitude (lag 60 min)", fontsize=10)
ax.legend(fontsize=8.5, loc="upper right")

plt.tight_layout()
out = os.path.join(ROOT, "figures", "fig9_latitude.png")
plt.savefig(out, dpi=300)
plt.close()
print("fig9_latitude (4-panel) saved ->", out)
