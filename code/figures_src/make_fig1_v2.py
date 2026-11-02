# -*- coding: utf-8 -*-
"""Fig 1 (resubmit v1, rev2): 3 panels stacked VERTICALLY (per Fang: uniform
layout), OMNI panel redrawn properly.

Fix vs rev1: (c) previously plotted the DAILY-resolution e10h rows
(~30 rows per month, n_min=1440) directly against month index -- 365 bars
piled on 12 x-positions, unreadable. Now aggregate to monthly means first and
show two clear bar groups: IMF Bz (100% observed) vs plasma params (natively
observed share, i.e. not filled by interpolation).

Responds to R3-1 (split Fig1, coastlines, OMNI coverage panel).
Output: resubmit_v1/figures/fig1_overview.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8,
                     "savefig.bbox": "tight"})

# old code: (mlat, glat, glon, valid_days, new code)
STATIONS = {
    "MHT": (43.9, 53.5, 122.3, 363, "OMOHE"), "MZL": (39.8, 49.5, 117.4, 364, "OMZLN"),
    "NAT": (34.6, 44.1, 125.0, 358, "ONOAN"), "SSL": (30.6, 40.3, 116.2, 358, "OCSSL"),
    "MLS": (25.0, 34.7, 118.5, 336, "OTANC"), "PXT": (21.0, 30.9, 103.9, 355, "OPIXI"),
    "JFT": (20.7, 30.5, 114.5, 344, "OYICH"), "HZT": (20.6, 30.25, 120.2, 362, "OHZXH"),
    "SYT": (17.4, 27.2, 111.5, 354, "OSHYA"), "ZQT": (13.4, 23.1, 112.5, 330, "OZHQI"),
    "SYS": (9.7, 19.5, 109.6, 343, "OLEDO"), "QZT": (9.3, 19.0, 109.8, 292, "OQIZH"),
    "LAT": (19.61, 29.1, 91.1, 349, "OLSCG"),
}

fig = plt.figure(figsize=(8.6, 10.2))
gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 0.85, 0.85], hspace=0.32)

# ---------------- (a) map with coastlines ----------------
axm = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
axm.set_extent([62, 150, 0, 62], crs=ccrs.PlateCarree())
axm.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f2efe9")
axm.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dceaf5")
axm.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.6, edgecolor="#555555")
axm.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.5, edgecolor="#999999",
                linestyle=":")
gl = axm.gridlines(draw_labels=True, lw=0.3, color="gray", alpha=0.5)
gl.top_labels = False; gl.right_labels = False

mlats = [v[0] for v in STATIONS.values()]
sc = axm.scatter([v[2] for v in STATIONS.values()], [v[1] for v in STATIONS.values()],
                 c=mlats, cmap="viridis", s=60, edgecolor="k", linewidth=0.6,
                 transform=ccrs.PlateCarree(), zorder=5)
LBL_OFF = {"OMOHE": (5, 5), "OMZLN": (5, 5), "ONOAN": (5, 5), "OCSSL": (5, 5),
           "OTANC": (5, 5), "OPIXI": (-30, 6), "OYICH": (9, -13),
           "OHZXH": (9, 4), "OSHYA": (-17, 7), "OZHQI": (6, 3),
           "OLEDO": (-42, -4), "OQIZH": (9, -10), "OLSCG": (5, 5)}
for st, (mla, gla, glo, days, nc) in STATIONS.items():
    dx, dy = LBL_OFF[nc]
    axm.annotate(nc, (glo, gla), fontsize=7.5, xytext=(dx, dy),
                 textcoords="offset points", transform=ccrs.PlateCarree(),
                 bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))
cb = plt.colorbar(sc, ax=axm, shrink=0.8, pad=0.02)
cb.set_label("Magnetic latitude (deg N)")
axm.set_title("(a) Station distribution (Chinese Meridian Project magnetometer chain)",
              fontsize=10)

# ---------------- (b) valid-day coverage ----------------
ax = fig.add_subplot(gs[1])
order = ["MHT", "MZL", "NAT", "SSL", "MLS", "PXT", "JFT", "HZT", "SYT", "ZQT",
         "LAT", "SYS", "QZT"]
days = [STATIONS[s][3] for s in order]
mlat = [STATIONS[s][0] for s in order]
ncodes = [STATIONS[s][4] for s in order]
ax.bar(range(len(order)), days, color="#1f77b4", alpha=0.85, width=0.65)
for i, d in enumerate(days):
    ax.text(i, d + 5, str(d), ha="center", fontsize=7)
ax.axhline(365, color="gray", ls=":", lw=0.8)
ax.set_xticks(range(len(order)))
ax.set_xticklabels([f"{nc}\n{mlat[i]:.0f}$^\\circ$" for i, nc in enumerate(ncodes)],
                   fontsize=7)
ax.set_ylabel("Valid days (of 365)")
ax.set_ylim(0, 420)
ax.set_title("(b) Station data coverage, 2015", fontsize=10)

# ---------------- (c) OMNI coverage, monthly aggregates ----------------
om = pd.read_csv(os.path.join(ROOT, "results_e10", "e10h_omni_native.csv"))
g = om.groupby("month").apply(
    lambda d: 100.0 * d["native_plasma"].sum() / d["n_min"].sum(),
    include_groups=False).rename("plasma_pct")
months = g.index.to_numpy()
plasma = g.to_numpy()
# Bz: bz_valid_pct is 100 for every day -> monthly value is 100
bz_monthly = om.groupby("month")["bz_valid_pct"].mean().to_numpy()

x = months.astype(float)
w = 0.38
ax = fig.add_subplot(gs[2])
ax.bar(x - w / 2, plasma, w, color="#4c72b0", alpha=0.9,
       label="Plasma parameters (natively observed)")
ax.bar(x + w / 2, bz_monthly, w, color="#dd8452", alpha=0.9,
       label="IMF $B_z$ (natively observed)")
ax.axhline(plasma.mean(), color="#4c72b0", ls="--", lw=1.0,
           label=f"Plasma annual mean ({plasma.mean():.0f}%)")
ax.set_xticks(range(1, 13))
ax.set_xticklabels([str(m) for m in range(1, 13)], fontsize=8)
ax.set_xlabel("Month (2015)")
ax.set_ylabel("OMNI 1-min records,\nnatively observed (%)")
ax.set_ylim(0, 112)
ax.set_xlim(0.4, 13.4)
ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
ax.set_title("(c) OMNI data coverage by month (1-min records)", fontsize=10)

plt.tight_layout()
out = os.path.join(ROOT, "figures", "fig1_overview.png")
plt.savefig(out, dpi=150)
plt.close()
print("Fig1 saved ->", out)
print("monthly plasma native %:", [f"{v:.1f}" for v in plasma])
print(f"annual mean: {plasma.mean():.1f}%  min {plasma.min():.1f}  max {plasma.max():.1f}")
