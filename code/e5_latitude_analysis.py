# -*- coding: utf-8 -*-
"""E5: 14 站磁纬链分析 — 基于 E3 结果（无需新计算）
坐标: 9 站取自 ZW station location.xlsx（含 mlat）；SYT 同表；
     QZT/SSL/ZQT 用公开地理坐标 + 本台站链一致的地磁偏移(-9.7°)估算（注明近似）；
     ZSZ（南极中山站）跨半球，单独标注，纬度分析用 |mlat| 时除外/单列。
"""
import glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"C:\Users\Lenovo\Desktop\2026_Entropy征稿\server_sync\results_e3"
E3_DIR = r"C:\Users\Lenovo\Desktop\2026_Entropy征稿\server_sync\results_e3"

# code: (mlat, glat, glon, source)
STATIONS = {
    "MHT": (43.9, 53.5, 122.3, "table"),
    "MZL": (39.8, 49.5, 117.4, "table"),
    "NAT": (34.6, 44.1, 125.0, "table"),
    "SSL": (30.6, 40.3, 116.2, "public-approx"),
    "MLS": (25.0, 34.7, 118.5, "table"),
    "PXT": (21.0, 30.9, 103.9, "table"),
    "JFT": (20.7, 30.5, 114.5, "table"),
    "HZT": (20.6, 30.25, 120.2, "table"),
    "SYT": (17.4, 27.2, 111.5, "table"),
    "ZQT": (13.4, 23.1, 112.5, "public-approx"),
    "SYS": (9.7, 19.5, 109.6, "table"),
    "QZT": (9.3, 19.0, 109.8, "public-approx"),
    "ZSZ": (-74.5, -69.4, 76.4, "public-approx"),
}

dfs = [pd.read_csv(f) for f in glob.glob(f"{E3_DIR}/sig_Pc5_5min_*.csv")]
df = pd.concat(dfs)
df = df[~df.station.isna()]
df["excess"] = df.TE_true - df.null_mean
df["mlat"] = df.station.map(lambda s: STATIONS.get(s, (np.nan,)*4)[0])
df["src"] = df.station.map(lambda s: STATIONS.get(s, (0, 0, 0, "?"))[3])
df = df.dropna(subset=["mlat"])
df.to_csv(f"{OUT}/e5_station_mlat.csv", index=False)
print("stations in E3:", sorted(df.station.unique()))

for lag_sel in [20, 60, 120]:
    d = df[(df.lag_min == lag_sel) & (df.param == "Flow_pressure") & (df.station != "ZSZ")]
    r, p = spearmanr(d.mlat, d.excess)
    rz, pz = spearmanr(d.mlat, d.z)
    print(f"Pdyn lag={lag_sel}: excess|mlat Spearman r={r:.2f} (p={p:.3f}), z|mlat r={rz:.2f} (p={pz:.3f}), n={len(d)}")

# --- 图 ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
colors = {20: "#1f77b4", 60: "#d62728", 120: "#2ca02c"}
# (a) excess TE vs mlat (Pdyn, 3 lags)
ax = axes[0]
for lag_sel in [20, 60, 120]:
    d = df[(df.lag_min == lag_sel) & (df.param == "Flow_pressure") & (df.station != "ZSZ")].groupby("station")[["mlat", "excess"]].median()
    ax.plot(d.mlat, d.excess, "o", color=colors[lag_sel], ms=7, label=f"lag {lag_sel} min")
ax.set_xlabel("Magnetic latitude (deg N)"); ax.set_ylabel("Excess TE (nats)")
ax.set_title("(a) Pdyn -> Pc5: latitude profile", fontsize=10); ax.legend(fontsize=8)
for st, (ml, *_ ) in STATIONS.items():
    if st != "ZSZ":
        ax.annotate(st, (ml, df[(df.station==st)&(df.param=="Flow_pressure")&(df.lag_min==60)].excess.median()),
                    fontsize=6, xytext=(0, 6), textcoords="offset points", ha="center")
# (b) z vs mlat (Pdyn lag 60)
ax = axes[1]
d = df[(df.lag_min == 60) & (df.param == "Flow_pressure")].groupby("station")[["mlat", "z"]].median()
d["src"] = df[(df.lag_min == 60) & (df.param == "Flow_pressure")].groupby("station")["src"].first()
for src, mk in [("table", "o"), ("public-approx", "s")]:
    dd = d[d.src == mk]
    ax.plot(dd.mlat, dd.z, mk, ms=8, label={"table": "table mlat", "public-approx": "approx mlat"}[src])
ax.axhline(2, color="gray", ls=":", lw=1)
ax.set_xlabel("Magnetic latitude (deg N)"); ax.set_ylabel("z (shift-null)")
ax.set_title("(b) Pdyn significance vs latitude (lag 60 min)", fontsize=10); ax.legend(fontsize=8)
for st in d.index:
    ax.annotate(st, (d.loc[st, "mlat"], d.loc[st, "z"]), fontsize=6, xytext=(0, 6), textcoords="offset points", ha="center")
# (c) peak-lag vs mlat (Pdyn)
ax = axes[2]
pk = df[df.param == "Flow_pressure"].loc[df[df.param == "Flow_pressure"].groupby("station")["TE_true"].idxmax()]
pk = pk[pk.station != "ZSZ"]
ax.plot(pk.mlat, pk.lag_min, "o", color="#9467bd", ms=8)
ax.set_xlabel("Magnetic latitude (deg N)"); ax.set_ylabel("Peak lag (min)")
ax.set_title("(c) Pdyn: peak-lag vs latitude", fontsize=10)
for st in pk.station:
    ax.annotate(st, (pk[pk.station==st].mlat.iloc[0], pk[pk.station==st].lag_min.iloc[0]),
                fontsize=6, xytext=(0, 6), textcoords="offset points", ha="center")
plt.suptitle("E5: latitude dependence of solar-wind -> Pc5 information flow (13 stations, 2015)", fontsize=11)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_e5_latitude.png", dpi=150)
print("fig saved")

# 汇总表
tab = df[df.param=="Flow_pressure"].groupby("station").agg(
    mlat=("mlat","median"), excess_lag20=("excess", lambda s: np.nan),
    z_lag60=("z", "median"))
d60 = df[(df.lag_min==60)&(df.param=="Flow_pressure")].set_index("station")
d20 = df[(df.lag_min==20)&(df.param=="Flow_pressure")].set_index("station")
tab["excess_lag20"] = d20["excess"]; tab["z_lag20"] = d20["z"]
tab["excess_lag60"] = d60["excess"]; tab["z_lag60"] = d60["z"]
tab["src"] = d60["src"]
print(tab.sort_values("mlat", ascending=False).round(4).to_string())
tab.to_csv(f"{OUT}/e5_summary_table.csv")
