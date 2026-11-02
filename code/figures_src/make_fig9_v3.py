# -*- coding: utf-8 -*-
"""Fig 9/10 (resubmit v1, rev3): latitude dependence, redesigned for clarity.

Changes vs rev2 (per Fang: reviewer's point still not visually resolved):
  - 3 panels -> 2 panels: the old (c) peak-lag panel carried no information
    (scatter noise, not referenced in the text) and is dropped.
  - Station-name labels are no longer printed for all 13 stations (they were
    the main source of clutter); only stations named in the text are labelled
    (OMOHE, OSHYA in panel a; OHZXH in panel b).
  - Same-station triplets still joined by thin grey lines (R3 request),
    per-lag OLS fits kept; the 20-min fit (the text's primary lag) is heavier.
  - In-panel annotation of the Spearman coefficient quoted in Section 4.6.
Output: resubmit_v1/figures/fig9_latitude.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = r"E:/project_ulf_omni/ground_pc5/server_sync/results_e3/e5_station_mlat.csv"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CODE_MAP = {"MHT": "OMOHE", "MZL": "OMZLN", "NAT": "ONOAN", "SSL": "OCSSL",
            "MLS": "OTANC", "PXT": "OPIXI", "JFT": "OYICH", "HZT": "OHZXH",
            "SYT": "OSHYA", "ZQT": "OZHQI", "LAT": "OLSCG", "SYS": "OLEDO",
            "QZT": "OQIZH"}
nw = lambda s: CODE_MAP.get(s, s)

df = pd.read_csv(CSV)
df = df[df.station != "ZSZ"].copy()
df["stn"] = df.station.map(nw)
LAG_COLORS = {20: "#1f77b4", 60: "#d62728", 120: "#2ca02c"}

plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.9,
                     "savefig.bbox": "tight"})
fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), gridspec_kw={"wspace": 0.24})

# ============ (a) excess TE vs latitude: triplets joined + per-lag fits ======
ax = axes[0]
med = (df[(df.param == "Flow_pressure") & (df.lag_min.isin([20, 60, 120]))]
       .groupby(["stn", "lag_min"])[["mlat", "excess"]].median().reset_index())
# thin grey lines joining the three lag points of each station (R3 request)
for st, g in med.groupby("stn"):
    g = g.sort_values("lag_min")
    ax.plot(g.mlat, g.excess, "-", color="#c9c9c9", lw=0.8, zorder=1)
# points + per-lag OLS fits (20-min = the text's primary lag -> heavier)
for lag in [20, 60, 120]:
    g = med[med.lag_min == lag].sort_values("mlat")
    ax.plot(g.mlat, g.excess, "o", color=LAG_COLORS[lag], ms=lag == 20 and 7 or 6,
            label=f"lag {lag} min", zorder=3)
    p = np.polyfit(g.mlat, g.excess, 1)
    xs = np.linspace(g.mlat.min(), g.mlat.max(), 30)
    ax.plot(xs, np.polyval(p, xs), "-", color=LAG_COLORS[lag],
            lw=2.2 if lag == 20 else 1.2, alpha=0.6, zorder=2)
# label only the two stations named in Section 4.6
for st, (dx, dy) in {"OMOHE": (8, -4), "OSHYA": (-38, -14)}.items():
    g = med[(med.stn == st) & (med.lag_min == 20)].iloc[0]
    ax.annotate(st, (g.mlat, g.excess), fontsize=9, xytext=(dx, dy),
                textcoords="offset points",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.5))
ax.text(0.03, 0.96, "Spearman $r_s=-0.59$\n($p=0.035$, $n=13$; 20-min excess)",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))
ax.set_xlabel("Magnetic latitude (deg N)")
ax.set_ylabel("$P_{dyn}$ excess TE (nats)")
ax.set_title("(a) Excess TE vs latitude (grey lines join each station's three lags)",
             fontsize=10)
ax.legend(fontsize=9, loc="lower left")

# ============ (b) surrogate z vs latitude (lag 60) ===========================
ax = axes[1]
d = (df[(df.lag_min == 60) & (df.param == "Flow_pressure")]
     .groupby("station").agg(mlat=("mlat", "median"), z=("z", "median"),
                             src=("src", "first")))
for src, mk, lb in [("table", "o", "project-table mlat"),
                    ("public-approx", "s", "approximate mlat")]:
    dd = d[d.src == src]
    ax.plot(dd.mlat, dd.z, mk, ms=9, mfc="#d62728" if src == "table" else "none",
            mec="#d62728", mew=1.4, label=lb, zorder=3)
p = np.polyfit(d.mlat, d.z, 1)
xs = np.linspace(d.mlat.min(), d.mlat.max(), 30)
ax.plot(xs, np.polyval(p, xs), "--", color="#555555", lw=1.6, alpha=0.8,
        label=f"OLS fit (slope {p[0]:+.3f} z/deg)")
ax.axhline(2, color="gray", ls=":", lw=1)
ax.text(d.mlat.max() + 0.3, 2.1, "z = 2", fontsize=8, color="gray")
# label only the data-quality suspect named in the text
g = d.loc["HZT"]
ax.annotate("OHZXH\n(susp.)", (g.mlat, g.z), fontsize=9, xytext=(6, -16),
            textcoords="offset points",
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.5))
ax.set_xlabel("Magnetic latitude (deg N)")
ax.set_ylabel("Surrogate $z$ (lag 60 min)")
ax.set_ylim(0, 8.4)
ax.set_title("(b) Significance vs latitude", fontsize=10)
ax.legend(fontsize=9, loc="upper right")

plt.tight_layout()
out = os.path.join(ROOT, "figures", "fig9_latitude.png")
plt.savefig(out, dpi=300)
plt.close()
print("fig9_latitude saved ->", out)
