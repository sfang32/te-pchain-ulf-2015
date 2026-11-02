# -*- coding: utf-8 -*-
"""Fig 9 (resubmit v1): latitude dependence, R3 revision — connect the three
lag-colour points of each station with a thin line and add per-colour linear
fits; add fit line in panel (b) as well.
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

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
colors = {20: "#1f77b4", 60: "#d62728", 120: "#2ca02c"}

# ---------------- (a) excess TE vs latitude, 3 lags, connected + fits ----------------
ax = axes[0]
for lag in [20, 60, 120]:
    d = df[(df.lag_min == lag) & (df.param == "Flow_pressure")].groupby("stn")[
        ["mlat", "excess"]].median().sort_values("mlat")
    ax.plot(d.mlat, d.excess, "o", color=colors[lag], ms=6.5, label=f"lag {lag} min",
            zorder=3)
    # per-colour linear fit
    p = np.polyfit(d.mlat, d.excess, 1)
    xs = np.linspace(d.mlat.min(), d.mlat.max(), 20)
    ax.plot(xs, np.polyval(p, xs), "-", color=colors[lag], lw=1.2, alpha=0.55,
            zorder=2)
    print(f"  (a) lag {lag}: slope = {p[0]:+.5f} nats/deg")
# connect same-station triplets
for st in sorted(df.stn.unique()):
    dd = df[(df.stn == st) & (df.param == "Flow_pressure") & (df.lag_min.isin([20, 60, 120]))]
    dd = dd.groupby("lag_min")[["mlat", "excess"]].median().sort_index()
    ax.plot(dd.mlat, dd.excess, "-", color="#bbbbbb", lw=0.6, zorder=1)
    ax.annotate(st, (dd.mlat.loc[60], dd.excess.loc[60]), fontsize=6,
                xytext=(0, 7), textcoords="offset points", ha="center")
ax.set_xlabel("Magnetic latitude (deg N)")
ax.set_ylabel("Excess TE (nats)")
ax.set_title("(a) $P_{dyn}\\to$ Pc5: latitude profile (lines join stations; fits per lag)",
             fontsize=9)
ax.legend(fontsize=8)

# ---------------- (b) z vs latitude (lag 60) with fit ----------------
ax = axes[1]
d = df[(df.lag_min == 60) & (df.param == "Flow_pressure")].groupby("station").agg(
    mlat=("mlat", "median"), z=("z", "median"), src=("src", "first"))
for src, mk in [("table", "o"), ("public-approx", "s")]:
    dd = d[d.src == src].copy()
    dd["stn"] = dd.index.map(nw)
    ax.plot(dd.mlat, dd.z, mk, ms=8,
            label={"table": "table mlat", "public-approx": "approx mlat"}[src])
p = np.polyfit(d.mlat, d.z, 1)
xs = np.linspace(d.mlat.min(), d.mlat.max(), 20)
ax.plot(xs, np.polyval(p, xs), "-", color="#d62728", lw=1.2, alpha=0.55)
print(f"  (b) z slope = {p[0]:+.4f} z/deg")
ax.axhline(2, color="gray", ls=":", lw=1)
ax.set_xlabel("Magnetic latitude (deg N)")
ax.set_ylabel("z (shift-null)")
ax.set_title("(b) $P_{dyn}$ significance vs latitude (lag 60 min)", fontsize=10)
ax.legend(fontsize=8)
for st in d.index:
    ax.annotate(nw(st), (d.loc[st, "mlat"], d.loc[st, "z"]), fontsize=6,
                xytext=(0, 6), textcoords="offset points", ha="center")

# ---------------- (c) peak-lag vs latitude ----------------
ax = axes[2]
pk = df[df.param == "Flow_pressure"].loc[
    df[df.param == "Flow_pressure"].groupby("station")["TE_true"].idxmax()].copy()
pk["stn"] = pk.station.map(nw)
ax.plot(pk.mlat, pk.lag_min, "o", color="#9467bd", ms=8)
ax.set_xlabel("Magnetic latitude (deg N)")
ax.set_ylabel("Peak lag (min)")
ax.set_title("(c) $P_{dyn}$: peak-lag vs latitude", fontsize=10)
for st in pk.station:
    ax.annotate(nw(st), (pk[pk.station == st].mlat.iloc[0],
                         pk[pk.station == st].lag_min.iloc[0]),
                fontsize=6, xytext=(0, 6), textcoords="offset points", ha="center")

plt.tight_layout()
out = os.path.join(ROOT, "figures", "fig9_latitude.png")
plt.savefig(out, dpi=300)
plt.close()
print("fig9 saved ->", out)
