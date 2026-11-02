# -*- coding: utf-8 -*-
"""E0: 数据准备 — 台站 1Hz -> 1-min 带限功率 (Pc3/Pc4/Pc5, Jacobs et al. 1964 口径)
Pc3: 10-45 s   -> 22.22-100 mHz
Pc4: 45-150 s  -> 6.67-22.22 mHz
Pc5: 150-600 s -> 1.67-6.67 mHz
输出: data/{station}_{year}_1min.csv.gz  (index=1min时间戳, 列=Pc3_H,Pc4_H,Pc5_H + OMNI参数)
用法: python e0_data_prep.py --year 2015 [--stations MHT,MLS ...] [--workers 8]
"""
import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from multiprocessing import Pool

ROOT = "/data/ssd-stor/fsf/workspaces/research_ulf"
STATION_DIR = f"{ROOT}/datas/dataset_pc/ZW"
OMNI_DIR = f"{ROOT}/submit_yuanxinxihua/data_omni"
OUT_DIR = f"{ROOT}/jb_te_jastp/data"

BANDS = {  # Hz
    "Pc3": (1 / 45, 1 / 10),    # 22.2-100 mHz (10-45 s)
    "Pc4": (1 / 150, 1 / 45),   # 6.67-22.2 mHz (45-150 s)
    "Pc5": (1 / 600, 1 / 150),  # 1.67-6.67 mHz (150-600 s)
}
OMNI_COLS = ["Bz_GSM", "Flow_speed", "Flow_pressure", "Electric_field",
             "AE-index", "AL-index", "AU-index", "SYM-H", "ASY-H"]
MIN_COVERAGE = 0.9  # 当日有效数据 <90% 则丢弃


def load_day_1hz(station: str, year: str, date: str) -> pd.DataFrame:
    """拼接某日 10-min 窗口 1Hz 数据, 返回 H 列 1s 重整序列"""
    month = date[5:7]
    files = sorted(glob.glob(f"{STATION_DIR}/{station}/{year}/10min/feature_1Hz/{month}/{date}*.csv"))
    if len(files) < 100:  # 全天 144 窗, 少于 100 个直接放弃
        return None
    dfs = [pd.read_csv(f, index_col=0, parse_dates=[0], usecols=[0, 1]) for f in files]
    ser = pd.concat(dfs).sort_index().iloc[:, 0]
    full_idx = pd.date_range(f"{date} 00:00:00", f"{date} 23:59:59", freq="1s")
    ser = ser.reindex(full_idx)
    coverage = ser.notna().mean()
    if coverage < MIN_COVERAGE:
        return None
    return ser.interpolate(limit=30).ffill().bfill(), coverage


def band_power_1min(h: np.ndarray, date: str) -> pd.DataFrame:
    """Butterworth 带通 + 平方 + 1-min 平均"""
    fs = 1.0
    out = {}
    for band, (lo, hi) in BANDS.items():
        b, a = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band")
        x = filtfilt(b, a, h - h.mean())
        s = pd.Series(x ** 2, index=pd.date_range(f"{date} 00:00:00", periods=len(x), freq="1s"))
        out[band] = s.resample("1min").mean()
    return pd.DataFrame(out)


def process_day(args):
    station, year, date = args
    try:
        res = load_day_1hz(station, year, date)
        if res is None:
            return None
        ser, coverage = res
        df = band_power_1min(ser.values.astype(float), date)
        df["coverage"] = coverage
        df["station"] = station
        return df
    except Exception as e:
        print(f"[WARN] {station} {date}: {e}", flush=True)
        return None


def run_station(station: str, year: str, workers: int):
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D").strftime("%Y-%m-%d")
    tasks = [(station, year, d) for d in dates]
    with Pool(workers) as pool:
        results = pool.map(process_day, tasks)
    results = [r for r in results if r is not None]
    if not results:
        print(f"[{station}] {year}: 无有效数据", flush=True)
        return
    df = pd.concat(results).sort_index()
    df = df[~df.index.duplicated()]

    # 合并 OMNI
    omni_files = sorted(glob.glob(f"{OMNI_DIR}/omni_{year}*.csv"))
    omni = pd.concat([pd.read_csv(f, index_col=0, parse_dates=[0]) for f in omni_files]).sort_index()
    omni = omni[~omni.index.duplicated()]
    for c in OMNI_COLS:
        if c not in omni.columns:
            omni[c] = np.nan
    df = df.join(omni[OMNI_COLS], how="inner")

    out = f"{OUT_DIR}/{station}_{year}_1min.csv.gz"
    df.to_csv(out, compression="gzip")
    n_days = df["coverage"].resample("D").first().notna().sum()
    print(f"[OK] {station} {year}: {len(df)} 分钟点, 有效天={n_days}/365, -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2015")
    ap.add_argument("--stations", default="MHT,MZL,NAT,MLS,JFT,HZT,PXT,LAT,SYS")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    for st in args.stations.split(","):
        run_station(st.strip(), args.year, args.workers)
    print("E0 DONE", flush=True)
