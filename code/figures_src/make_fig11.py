# -*- coding: utf-8 -*-
"""Fig 11 (resubmit v1): E10 control analyses — (a) LT day/night excess TE,
(b) absolute vs variability of Pdyn, (c) AE-conditioned z.
Output: resubmit_v1/figures/fig11_lt_variability.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R = os.path.join(ROOT, "results_e10")

CODE_MAP = {"MHT": "OMOHE", "MZL": "OMZLN", "NAT": "ONOAN", "SSL": "OCSSL",
            "MLS": "OTANC", "PXT": "OPIXI", "JFT": "OYICH", "HZT": "OHZXH",
            "SYT": "OSHYA", "ZQT": "OZHQI", "LAT": "OLSCG", "SYS": "OLEDO",
            "QZT": "OQIZH"}
MLAT = {"MHT": 43.9, "MZL": 39.8, "NAT": 34.6, "SSL": 30.6, "MLS": 25.0,
        "PXT": 21.0, "JFT": 20.7, "HZT": 20.6, "SYT": 17.4, "ZQT": 13.4,
        "SYS": 9.7, "QZT": 9.3, "LAT": 19.61}
nw = lambda s: CODE_MAP.get(s, s)

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8,
                     "figure.dpi": 300, "savefig.bbox": "tight"})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

# ---------------- (a) LT split: day vs night excess TE (lag 20) ----------------
ax = axes[0]
lt = pd.read_csv(os.path.join(R, "e10c_lt_split.csv"))
lt = lt[lt.lag_min == 20].copy()
lt["mlat"] = lt.station.map(MLAT)
day = lt[lt.group == "day"].sort_values("mlat").reset_index(drop=True)
night = lt[lt.group == "night"].set_index("station").loc[day.station].reset_index()
x = np.arange(len(day))
ed = day.TE_true - day.null_mean
en = night.TE_true - night.null_mean
sd = 1.96 * day.null_std
sn = 1.96 * night.null_std
ax.errorbar(x - 0.15, ed, yerr=sd, fmt="o", ms=5.5, color="#d62728",
            lw=1.1, capsize=2, label="dayside 06–18 LT")
ax.errorbar(x + 0.15, en, yerr=sn, fmt="o", ms=5.5, color="#1f77b4",
            lw=1.1, capsize=2, label="nightside 18–06 LT")
for i in range(len(day)):                      # connect same-station pairs
    ax.plot([x[i] - 0.15, x[i] + 0.15], [ed[i], en[i]], "-", color="#bbbbbb",
            lw=0.7, zorder=1)
ax.axhline(0, color="gray", lw=0.6, ls=":")
ax.set_xticks(x)
ax.set_xticklabels([nw(s) for s in day.station], rotation=60, fontsize=6.5)
ax.set_ylabel("Excess TE at 20 min (nats)")
ax.set_title("(a) Dayside vs nightside (13 stations, $P_{dyn}$)", fontsize=10)
ax.legend(fontsize=8)

# ---------------- (b) absolute vs variability source ----------------
ax = axes[1]
var = pd.read_csv(os.path.join(R, "e10d_variability.csv"))
main = pd.read_csv(
    r"E:/project_ulf_omni/ground_pc5/server_sync/results_e3/e5_station_mlat.csv")
m3 = main[(main.param == "Flow_pressure") & (main.lag_min.isin([20, 45]))
          & (main.station.isin(["MHT", "PXT", "SYS"]))]
stations = ["MHT", "PXT", "SYS"]
x = np.arange(len(stations))
w = 0.19
combos = [("abs", 20, "#7f7f7f", "absolute $P_{dyn}$, 20 min"),
          ("var", 20, "#d62728", "variability $P_{dyn}^{\\sigma30}$, 20 min"),
          ("abs", 45, "#bdbdbd", "absolute $P_{dyn}$, 45 min"),
          ("var", 45, "#ff9896", "variability $P_{dyn}^{\\sigma30}$, 45 min")]
for k, (kind, lag, c, lab) in enumerate(combos):
    if kind == "abs":
        d = m3[m3.lag_min == lag].set_index("station").loc[stations]
        y = d.excess.values
        e = 1.96 * d.null_std.values
    else:
        d = var[(var.source == "Pdyn_std30") & (var.lag_min == lag)].set_index(
            "station").loc[stations]
        y = d.TE_true.values - d.null_mean.values
        e = 1.96 * d.null_std.values
    ax.bar(x + (k - 1.5) * w, y, width=w * 0.92, color=c, label=lab)
    ax.errorbar(x + (k - 1.5) * w, y, yerr=e, fmt="none", ecolor="k", lw=0.8,
                capsize=2)
ax.axhline(0, color="gray", lw=0.6)
ax.set_xticks(x)
ax.set_xticklabels([nw(s) for s in stations])
ax.set_ylabel("Excess TE (nats)")
ax.set_title("(b) Absolute vs 30-min-variability $P_{dyn}$ source", fontsize=10)
ax.legend(fontsize=7)

# ---------------- (c) AE conditioning (high-lat stations, lag 45) ----------------
ax = axes[2]
ae = pd.read_csv(os.path.join(R, "e10e_ae_conditioned.csv"))
ae = ae[ae.lag_min == 45].copy()
hl = ["MHT", "MZL", "NAT"]
ae = ae.set_index("station").loc[hl]
x = np.arange(len(hl))
w = 0.35
hi = ae.loc[ae.group == "AE_high"] if "AE_high" in ae.group.values else ae[ae.group == ae.group.unique()[0]]
lo = ae.loc[ae.group != hi.group.iloc[0]] if False else ae[ae.group != ae.group.unique()[0]]
# robust: explicit group columns
piv = ae.reset_index().pivot_table(index="station", columns="group", values="z")
piv = piv.loc[hl]
ax.bar(x - w/2, piv["AE_high"], width=w, color="#d62728", alpha=0.9,
       label="high AE (upper tercile)")
ax.bar(x + w/2, piv["AE_low"], width=w, color="#9ecae1", alpha=0.9,
       label="low AE (lower tercile)")
for i, st in enumerate(hl):                    # value labels
    ax.text(i - w/2, piv.loc[st, "AE_high"] + 0.08, f"{piv.loc[st, 'AE_high']:.1f}",
            ha="center", fontsize=7)
    ax.text(i + w/2, piv.loc[st, "AE_low"] + 0.08, f"{piv.loc[st, 'AE_low']:.1f}",
            ha="center", fontsize=7)
ax.axhline(2, color="gray", ls=":", lw=1)
ax.text(len(hl) - 0.45, 2.15, "z = 2", fontsize=7, color="gray")
ax.set_xticks(x)
ax.set_xticklabels([nw(s) for s in hl])
ax.set_ylabel("Surrogate z (45 min)")
ax.set_ylim(0, max(6.2, piv.values.max() * 1.15))
ax.set_title("(c) AE conditioning, high-latitude stations", fontsize=10)
ax.legend(fontsize=8)

plt.tight_layout()
out = os.path.join(ROOT, "figures", "fig11_lt_variability.png")
plt.savefig(out, dpi=300)
plt.close()
print("fig11 saved ->", out)
