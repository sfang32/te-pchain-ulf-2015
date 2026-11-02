# -*- coding: utf-8 -*-
"""E0c: SuperMAG 全球站 1-min 地磁 -> 1-min 带限功率 (并入 jb_te_jastp 流水线)

并入目标: 把 SuperMAG 全球网数据转成与 e0_data_prep.py 同构的
  {station}_{year}_1min.csv.gz (列 Pc3,Pc4,Pc5,coverage,station),
使服务器侧 E2/E3/E5 的 station 列表直接加入全球站即可复用,
堵"只用子午工程(东亚扇区, 磁纬 9-44N)局部代表性"的审稿质疑。

频段口径: Jacobs et al. 1964 IAGA 标准 (与 e0 完全一致)
  Pc3: 10-45 s  -> 22.2-100 mHz   (注意: SuperMAG 1-min 采样 Nyquist=120s, Pc3 受混叠,
                                   全球对比以 Pc4/Pc5 为主, Pc3 列保留但论文中标注不可靠)
  Pc4: 45-150 s -> 6.67-22.2 mHz
  Pc5: 150-600 s-> 1.67-6.67 mHz

分量选择: SuperMAG 返回 N/E/Z (地理坐标系, nT). 用 N(地理北向) 近似水平 H,
  与子午工程 E0 的 H 分量对齐 (磁偏角差异在频段分析中量级可比).

OMNI: 本脚本不合并 OMNI (服务器 jb_te_jastp/data_omni 已有).
  上传服务器后, 用与 e0 相同的 join 逻辑合并 OMNI 列即可.

用法(小批量验证):
  python e0c_supermag_prep.py --year 2015 --stations CLF,TSU,BRW --months 3 --plot
用法(全年批量, 24 站):
  python e0c_supermag_prep.py --year 2015 \
      --stations-file ../../outputs/supermag_global/selected_stations.csv
"""
import argparse
import calendar
import glob
import os
import sys
from datetime import date, timedelta
import ssl
import urllib.request

# --- monkey-patch: supermag_api 默认强制 SSL 校验, 本环境代理自签证书需跳过 ---
_orig = urllib.request.urlopen
def _patched(url, *a, **k):
    k.pop("cafile", None)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return _orig(url, context=ctx, *a, **k)
urllib.request.urlopen = _patched

ENV = r"C:/Users/Lenovo/.workbuddy/binaries/python/envs/default/Lib/site-packages"
if ENV not in sys.path:
    sys.path.insert(0, ENV)
from supermag_api import supermag_api as sm

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

LOGON = "fangsf"
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "data_supermag")
os.makedirs(OUT_DIR, exist_ok=True)

# Jacobs 1964 bands (Hz)
# SuperMAG 1-min: Nyquist = 120 s (0.00833 Hz).
#   Pc5 (150-600 s, 1.67-6.67 mHz): 完全在 Nyquist 内 -> 可靠, 全球对比主频段.
#   Pc4 (45-150 s): 仅 120-150 s 段可靠, 整体混叠 -> 不在此计算, 留 CARISMA 20Hz 补.
#   Pc3 (10-45 s)  : 完全超出 Nyquist -> 混叠不可用.
# 故 E0c 只产 Pc5 (真实), Pc4/Pc3 列填 NaN 以保持与 e0 同构 (列集一致, E2/E3/E5 选 Pc5 即可).
BANDS = {
    "Pc5": (1 / 600, 1 / 150),  # 1.67-6.67 mHz (150-600 s)
}
FS = 1.0 / 60.0  # SuperMAG 1-min
MIN_COVERAGE = 0.8


def pull_month(station, year, month):
    """拉取某站某月整月 1-min 数据, 返回 (status, df)"""
    start = [year, month, 1, 0, 0, 0]
    ext = calendar.monthrange(year, month)[1] * 86400
    st, df = sm.SuperMAGGetData(LOGON, start, ext, "all", station)
    if not st or (hasattr(df, "__len__") and len(df) == 0):
        return False, None
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return True, df


def extract_n(df):
    """取 N(地理北向≈水平H) 1-min 序列, 返回 (Series[utc], glat)"""
    if "N" not in df.columns:
        return None, np.nan
    col = df["N"]
    if isinstance(col.iloc[0], dict):
        n = col.apply(lambda d: d.get("nez") if isinstance(d, dict) else None)
    else:
        n = col
    t = pd.to_datetime(df["tval"], unit="s", utc=True)
    s = pd.Series(pd.to_numeric(n, errors="coerce").values.astype(float), index=t).sort_index()
    glat = float(pd.to_numeric(df["glat"], errors="coerce").dropna().median()) \
        if "glat" in df.columns else np.nan
    return s, glat


def band_power_daily(series_1min, date):
    """对单日 1-min 序列做 Butterworth 带通 + 平方 + 1-min 均值 (与 e0 同构)"""
    out = {}
    x0 = series_1min.values.astype(float)
    x0 = x0 - np.nanmean(x0)
    for band, (lo, hi) in BANDS.items():
        b, a = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="band")
        x = filtfilt(b, a, x0)
        s = pd.Series(x ** 2, index=series_1min.index)
        out[band] = s.resample("1min").mean()
    return pd.DataFrame(out)


def run_station(station, year, max_weeks=None):
    """按周分块拉取 (每块 7 天, 规避整月 extent 对部分站失败), 逐日独立滤波."""
    frames = []
    glat = np.nan
    d = date(year, 1, 1)
    end = date(year, 12, 31)
    wk = 0
    while d <= end:
        if max_weeks is not None and wk >= max_weeks:
            break
        ndays = min(7, (end - d).days + 1)
        start = [d.year, d.month, d.day, 0, 0, 0]
        ext = ndays * 86400
        ok, df = sm.SuperMAGGetData(LOGON, start, ext, "all", station)
        if ok and df is not None and len(df) > 0:
            ser, g = extract_n(df)
            if ser is not None and len(ser) >= 100:
                if not np.isnan(g):
                    glat = g
                # 按日拆分, 逐日独立滤波 (与 e0 一致)
                for day, day_ser in ser.groupby(ser.index.date):
                    day_ser = day_ser.dropna()
                    if len(day_ser) < 600:  # <约10h 放弃当日
                        continue
                    dp = band_power_daily(day_ser, str(day))
                    dp["coverage"] = day_ser.notna().mean()
                    frames.append(dp)
            else:
                print(f"  [{station}] {d}: N 分量缺失/过短", flush=True)
        else:
            print(f"  [{station}] {d}: 拉取失败(status={ok})", flush=True)
        d = d + timedelta(days=ndays)
        wk += 1
    if not frames:
        print(f"[{station}] {year}: 无有效数据", flush=True)
        return None
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated()]
    df["station"] = station
    df["glat"] = glat
    # 保持与 e0_data_prep.py 同构的列集 (Pc3/Pc4 因 1-min 混叠留空)
    for miss in ("Pc3", "Pc4"):
        if miss not in df.columns:
            df[miss] = np.nan
    out = os.path.join(OUT_DIR, f"{station}_{year}_1min_supermag.csv.gz")
    df.to_csv(out, compression="gzip")
    n_days = df["coverage"].resample("D").first().notna().sum()
    print(f"[OK] {station} {year}: {len(df)} 分钟点, 有效天~{n_days}, glat≈{glat:.1f}°N -> {out}", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2015)
    ap.add_argument("--stations", default="CLF,TSU,BRW")
    ap.add_argument("--stations-file", default=None, help="csv 含 code 列 (默认用 outputs/supermag_global/selected_stations.csv)")
    ap.add_argument("--max-weeks", type=int, default=None, help="调试用: 仅拉前 N 周 (默认全年)")
    ap.add_argument("--plot", action="store_true", help="画全球纬度剖面图")
    args = ap.parse_args()

    if args.stations_file:
        sdf = pd.read_csv(args.stations_file)
        stations = sdf["code"].tolist()
    else:
        stations = [s.strip() for s in args.stations.split(",")]

    print(f"=== E0c: SuperMAG 全球站 {args.year} 并入预处理 ({len(stations)} 站) ===", flush=True)
    collected = {}
    for st in stations:
        df = run_station(st, args.year, args.max_weeks)
        if df is not None:
            collected[st] = df

    if args.plot and collected:
        plot_profile(collected)

    print("E0c DONE", flush=True)


def plot_profile(collected):
    """画 Pc5(主) / Pc4 功率月均 vs 地理纬度, 叠加子午已有 Pc5 active_mean 作量级参考"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for st, df in collected.items():
        glat = df["glat"].dropna()
        glat = float(glat.median()) if len(glat) else np.nan
        rows.append({
            "station": st, "glat": glat,
            "Pc5": df["Pc5"].mean(), "Pc4": df["Pc4"].mean(),
            "Pc3": df["Pc3"].mean(),
        })
    sdf = pd.DataFrame(rows).dropna(subset=["glat"]).sort_values("glat")

    # 子午参考: results/latitude_dependence.csv 的 Pc5 active_mean
    ref_path = os.path.join(ROOT, "..", "results", "latitude_dependence.csv")
    ref = None
    if os.path.exists(ref_path):
        r = pd.read_csv(ref_path)
        r = r[r.band == "Pc5"]
        # E5 台站磁纬
        mlat = {"HZT":20.6,"JFT":20.7,"LAT":19.6,"MHT":43.9,"MLS":25.0,"MZL":39.8,
                "NAT":34.6,"PXT":21.0,"SYS":9.7,"SSL":30.6,"SYT":17.4,"QZT":9.3,"ZQT":13.4}
        r["mlat"] = r.station.map(mlat)
        ref = r.dropna(subset=["mlat", "active_mean"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sdf.glat, sdf.Pc5, s=70, c="#d62728", label="SuperMAG (global) Pc5 月均", zorder=3)
    for _, row in sdf.iterrows():
        ax.annotate(row.station, (row.glat, row.Pc5), fontsize=6, xytext=(0,5),
                    textcoords="offset points", ha="center")
    if ref is not None:
        ax.scatter(ref.mlat, ref.active_mean, s=60, c="#1f77b4", marker="^",
                   label="子午工程 Pc5 active-mean (量级参考)", zorder=3)
        for _, row in ref.iterrows():
            ax.annotate(row.station, (row.mlat, row.active_mean), fontsize=6, xytext=(0,5),
                        textcoords="offset points", ha="center")
    ax.set_xlabel("Geographic latitude (deg)")
    ax.set_ylabel("Pc5 band power (nT$^2$, monthly/active mean)")
    ax.set_title(f"E0c global latitude profile — SuperMAG (global) vs Meridian (magnitude check)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_supermag_latprofile.png")
    fig.savefig(out, dpi=150)
    print(f"[plot] -> {out}", flush=True)


if __name__ == "__main__":
    main()
