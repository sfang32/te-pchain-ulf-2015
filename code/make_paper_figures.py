# -*- coding: utf-8 -*-
"""论文出版图: Fig1 (数据总览), Fig2 (方法验证), Fig4 (z 热力图)
运行于服务器（数据与 E3 结果均在本地盘）。输出 results/paper_figures/*.png (300dpi)
"""
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/data/ssd-stor/fsf/workspaces/research_ulf/jb_te_jastp"
OUT = f"{ROOT}/results/paper_figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "figure.dpi": 300,
                     "savefig.bbox": "tight"})

STATIONS = {
    "MHT": (43.9, 53.5, 122.3, 363), "MZL": (39.8, 49.5, 117.4, 364),
    "NAT": (34.6, 44.1, 125.0, 358), "SSL": (30.6, 40.3, 116.2, 358),
    "MLS": (25.0, 34.7, 118.5, 336), "PXT": (21.0, 30.9, 103.9, 355),
    "JFT": (20.7, 30.5, 114.5, 344), "HZT": (20.6, 30.25, 120.2, 362),
    "SYT": (17.4, 27.2, 111.5, 354), "ZQT": (13.4, 23.1, 112.5, 330),
    "SYS": (9.7, 19.5, 109.6, 343), "QZT": (9.3, 19.0, 109.8, 292),
    "LAT": (19.61, 29.1, 91.1, 349),
}
PCOLORS = {"Bz_GSM": "#1f77b4", "Electric_field": "#9467bd",
           "Flow_pressure": "#d62728", "Flow_speed": "#2ca02c"}
PNAMES = {"Bz_GSM": "IMF $B_z$", "Electric_field": "E-field $E_y$",
          "Flow_pressure": "Dynamic pressure $P_{dyn}$", "Flow_speed": "Flow speed $V_{sw}$"}

# ============ Fig 1: data overview ============
fig = plt.figure(figsize=(13, 6.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])

axm = fig.add_subplot(gs[0, 0])
for st, (mla, gla, glo, days) in STATIONS.items():
    axm.plot(glo, gla, "o", color="#1f77b4", ms=7)
    axm.annotate(st, (glo, gla), fontsize=6.5, xytext=(4, 4), textcoords="offset points")
axm.set_xlim(85, 135); axm.set_ylim(5, 60)
axm.set_xlabel("Geographic longitude (deg)"); axm.set_ylabel("Geographic latitude (deg)")
axm.set_title("(a) Station distribution", fontsize=9)

day = "2015-03-17"
ax1 = fig.add_subplot(gs[0, 1:]); ax2 = ax1.twinx()
for st, c in [("MHT", "#d62728"), ("SYS", "#1f77b4")]:
    d = pd.read_csv(f"{ROOT}/data/{st}_2015_1min.csv.gz", index_col=0, parse_dates=[0])
    d = d.loc[day]
    ax1.plot(d.index, d["Pc5"], color=c, lw=0.8, label=f"{st} Pc5 power")
ax1.set_yscale("log"); ax1.set_ylabel("Pc5 band power (nT$^2$, log)")
omni_cols = [("Bz_GSM", "#2ca02c", "Bz (nT)"), ("Flow_pressure", "#9467bd", "$P_{dyn}$ (nPa)")]
for col, c, lab in omni_cols:
    ax2.plot(day_df.index if (day_df := pd.read_csv(f"{ROOT}/data/MHT_2015_1min.csv.gz", index_col=0, parse_dates=[0]).loc[day]) is not None else None,
             day_df[col], color=c, lw=0.8, label=lab)
ax3 = ax1.twinx(); ax3.spines["right"].set_position(("outward", 42))
ax3.plot(day_df.index, -day_df["SYM-H"], color="k", lw=0.8, label="-SYM-H")
ax3.set_ylabel("$-$SYM-H (nT)")
ax1.set_title(f"(b) St. Patrick storm day {day}: ground Pc5 power vs solar wind", fontsize=9)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
h3, l3 = ax3.get_legend_handles_labels()
ax1.legend(h1 + h2 + h3, l1 + l2 + l3, fontsize=6.5, loc="upper left", ncol=2)
ax1.set_xlabel("Time (UT)")

ax = fig.add_subplot(gs[1, :])
order = ["MHT", "MZL", "NAT", "SSL", "MLS", "PXT", "JFT", "HZT", "SYT", "ZQT", "LAT", "SYS", "QZT"]
mlats = [STATIONS[s][0] for s in order]
days = [STATIONS[s][3] for s in order]
ax.bar(range(len(order)), days, color="#1f77b4", alpha=0.8)
for i, (d, s) in enumerate(zip(days, order)):
    ax.text(i, d + 2, s, ha="center", fontsize=6.5)
ax.axhline(365, color="gray", ls=":", lw=0.8)
ax.set_xticks(range(len(order))); ax.set_xticklabels([f"{mlats[i]:.0f}" for i in range(len(order))])
ax.set_xlabel("Magnetic latitude (deg N)"); ax.set_ylabel("Valid days (of 365)")
ax.set_title("(c) Data coverage by station", fontsize=9)
ax.set_ylim(0, 400)
plt.savefig(f"{OUT}/fig1_overview.png"); plt.close()
print("Fig1 done", flush=True)

# ============ Fig 2: validation & null calibration ============
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
ax = axes[0]
ax.bar([0, 1], [0.0585, -0.0003], color=["#d62728", "#7f7f7f"], width=0.5)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks([0, 1]); ax.set_xticklabels(["TE(X$\\to$Y)", "TE(Y$\\to$X)"])
ax.set_ylabel("TE (nats)"); ax.set_title("(a) Coupled-AR benchmark (JIDT)", fontsize=9)
ax = axes[1]
nulls = {"iid": [0.0224, 0.0265], "shift (dest)": [0.0179, 0.0241]}
x = np.arange(2); w = 0.35
for i, (k, v) in enumerate(nulls.items()):
    ax.bar(x + (i - 0.5) * w, v, w, label=k, color=["#1f77b4", "#ff7f0e"][i])
ax.set_xticks(x); ax.set_xticklabels(["p95", "p99"])
ax.set_ylabel("Null percentile (nats)"); ax.set_title("(b) Null calibration (n=2000)", fontsize=9)
ax.legend(fontsize=8)
ax = axes[2]
vals = [0.0224, 0.0179]
ax.bar([0, 1], vals, color=["#1f77b4", "#ff7f0e"], width=0.5)
ax.axhline(0.0060, color="#d62728", lw=1.5)
ax.text(0.5, 0.0085, "typical Pdyn excess\n(0.006 nats)", color="#d62728", fontsize=7, ha="center")
ax.set_xticks([0, 1]); ax.set_xticklabels(["iid null p95", "shift null p95"])
ax.set_ylabel("nats"); ax.set_title("(c) Scale contrast: excess vs null", fontsize=9)
plt.savefig(f"{OUT}/fig2_validation.png"); plt.close()
print("Fig2 done", flush=True)

# ============ Fig 4: z heatmap ============
e3 = pd.concat([pd.read_csv(f) for f in glob.glob(f"{ROOT}/results/e3/sig_Pc5_5min_*.csv")])
order = ["MHT", "MZL", "NAT", "SSL", "MLS", "PXT", "JFT", "HZT", "SYT", "ZQT", "LAT", "SYS", "QZT"]
fig, axes = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True)
for ax, p in zip(axes, ["Flow_pressure", "Bz_GSM"]):
    d = e3[(e3.param == p)].copy()
    d["ord"] = d.station.map({s: i for i, s in enumerate(order)})
    d = d.dropna(subset=["ord"])
    piv = d.pivot_table(index="station", columns="lag_min", values="z").reindex(order)
    im = ax.pcolormesh(piv.columns, np.arange(len(piv)), piv.values, cmap="RdBu_r", vmin=-4, vmax=8)
    ax.set_yticks(np.arange(len(piv)) + 0.5); ax.set_yticklabels(piv.index, fontsize=7)
    ax.set_ylabel("Station")
    ax.set_title(f"{PNAMES[p]} -> Pc5", fontsize=9)
    cb = plt.colorbar(im, ax=ax, pad=0.01); cb.set_label("z (shift-null)")
axes[-1].set_xlabel("Lag (min)")
plt.suptitle("Fig 4: significance of information flow, station x lag", fontsize=11)
plt.savefig(f"{OUT}/fig4_z_heatmap.png"); plt.close()
print("Fig4 done", flush=True)
