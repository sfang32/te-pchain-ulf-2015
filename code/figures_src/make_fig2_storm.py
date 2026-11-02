# -*- coding: utf-8 -*-
"""Fig 2 (resubmit v1): St. Patrick's Day storm in separate panels (R3-1b).
(a) IMF Bz + dynamic pressure; (b) Pc5 band power at OMOHE/OLEDO; (c) -SYM-H
with SYM-H <= -50 nT storm condition shaded.
Output: resubmit_v1/figures/fig2_storm_overview.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATA = r"E:/project_ulf_omni/ground_pc5/server_sync/data"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8,
                     "figure.dpi": 300, "savefig.bbox": "tight"})

day = "2015-03-17"
d1 = pd.read_csv(f"{DATA}/MHT_2015_1min.csv.gz", index_col=0, parse_dates=[0])
d2 = pd.read_csv(f"{DATA}/SYS_2015_1min.csv.gz", index_col=0, parse_dates=[0])
d1 = d1[~d1.index.duplicated()].sort_index().loc[day]
d2 = d2[~d2.index.duplicated()].sort_index().loc[day]

fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True,
                         gridspec_kw={"height_ratios": [1, 1, 0.8]})

# (a) Bz + Pdyn
ax = axes[0]
ax.plot(d1.index, d1["Bz_GSM"], color="#2ca02c", lw=0.9, label="IMF $B_{z,\\mathrm{GSM}}$ (nT)")
ax.axhline(0, color="gray", lw=0.5, ls=":")
ax.set_ylabel("$B_z$ (nT)", color="#2ca02c")
ax.tick_params(axis="y", labelcolor="#2ca02c")
ax2 = ax.twinx()
ax2.plot(d1.index, d1["Flow_pressure"], color="#9467bd", lw=0.9,
         label="$P_{dyn}$ (nPa)")
ax2.set_ylabel("$P_{dyn}$ (nPa)", color="#9467bd")
ax2.tick_params(axis="y", labelcolor="#9467bd")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left", ncol=2)
ax.set_title("(a) Solar wind drivers (1-min OMNI)", fontsize=10)

# (b) Pc5 band power
ax = axes[1]
ax.plot(d1.index, d1["Pc5"], color="#d62728", lw=0.9, label="OMOHE (43.9$^\\circ$N mlat)")
ax.plot(d2.index, d2["Pc5"], color="#1f77b4", lw=0.9, label="OLEDO (9.7$^\\circ$N mlat)")
ax.set_yscale("log")
ax.set_ylabel("Pc5 band power (nT$^2$)")
ax.legend(fontsize=8, loc="upper right")
ax.set_title("(b) Ground Pc5 band power", fontsize=10)

# (c) -SYM-H with storm shading
ax = axes[2]
shade = d1["SYM-H"] <= -50
ax.fill_between(d1.index, 0, 260, where=shade, color="#ffcccc", alpha=0.6,
                label="SYM-H $\\leq$ $-$50 nT")
ax.plot(d1.index, -d1["SYM-H"], color="k", lw=1.0, label="$-$SYM-H (nT)")
ax.set_ylabel("$-$SYM-H (nT)")
ax.set_xlabel("Time (UT), 17 March 2015")
ax.set_ylim(0, 260)
ax.legend(fontsize=8, loc="upper left")
ax.set_title("(c) Geomagnetic storm condition", fontsize=10)

import matplotlib.dates as mdates
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=3))

plt.tight_layout()
out = os.path.join(ROOT, "figures", "fig2_storm_overview.png")
plt.savefig(out)
plt.close()
print("Fig2 saved ->", out)
