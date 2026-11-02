# -*- coding: utf-8 -*-
"""E0b: 从 YLK 原始 FGM .dat（H/Z/D/T 四分量, 1Hz, 每日一文件）计算 1-min 带限功率
数据根: /data/ssd-stor/YLK/YLK-BUFFER/NSSC-FSF/B015/MERIDIAN/Phase1/FGM/{站}/{年}/*.dat
输出与 e0_data_prep.py 一致: jb_te_jastp/data/{station}_{year}_1min.csv.gz
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from multiprocessing import Pool
from datetime import timedelta

YLK_ROOT = "/data/ssd-stor/YLK/YLK-BUFFER/NSSC-FSF/B015/MERIDIAN/Phase1/FGM"
OMNI_DIR = "/data/ssd-stor/fsf/workspaces/research_ulf/submit_yuanxinxinxihua/data_omni"
OMNI_DIR = "/data/ssd-stor/fsf/workspaces/research_ulf/submit_yuanxinxihua/data_omni"
OUT_DIR = "/data/ssd-stor/fsf/workspaces/research_ulf/jb_te_jastp/data"

BANDS = {  # Hz, Jacobs et al. 1964
    "Pc3": (1 / 45, 1 / 10),
    "Pc4": (1 / 150, 1 / 45),
    "Pc5": (1 / 600, 1 / 150),
}
OMNI_COLS = ["Bz_GSM", "Flow_speed", "Flow_pressure", "Electric_field",
             "AE-index", "AL-index", "AU-index", "SYM-H", "ASY-H"]
MIN_COVERAGE = 0.9


def load_dat(path: str):
    """解析单日 .dat -> 1s H 序列 (float32, 不用 float16 以避免量化噪声)"""
    with open(path, "r") as f:
        line = f.readline()
    parts = [p for p in line.split(" ") if p not in ("", "\n")]
    vals = parts[9:]
    n = len(vals) // 4
    if n < 80000:
        return None, 0.0
    arr = np.array(vals[: n * 4], dtype=np.float32).reshape(-1, 4)
    h = arr[:, 0]
    start = pd.to_datetime(os.path.basename(path).split(".")[0].split("_")[-1])
    idx = pd.date_range(start, periods=n, freq="1s")
    s = pd.Series(h, index=idx)
    full = pd.date_range(f"{idx[0].date()} 00:00:00", f"{idx[0].date()} 23:59:59", freq="1s")
    s = s.reindex(full)
    cov = s.notna().mean()
    if cov < MIN_COVERAGE:
        return None, cov
    return s.interpolate(limit=30).ffill().bfill(), cov


def band_power_1min(h: np.ndarray, date: str) -> pd.DataFrame:
    out = {}
    for band, (lo, hi) in BANDS.items():
        b, a = butter(4, [lo * 2, hi * 2], btype="band")
        x = filtfilt(b, a, h - h.mean())
        s = pd.Series(x ** 2, index=pd.date_range(f"{date} 00:00:00", periods=len(x), freq="1s"))
        out[band] = s.resample("1min").mean()
    return pd.DataFrame(out)


def process_day(args):
    station, year, path = args
    try:
        ser, cov = load_dat(path)
        if ser is None:
            return None
        date = str(ser.index[0].date())
        df = band_power_1min(ser.values.astype(float), date)
        df["coverage"] = cov
        return df
    except Exception as e:
        print(f"[WARN] {path}: {e}", flush=True)
        return None


def run_station(station: str, year: str, workers: int):
    files = sorted(glob.glob(f"{YLK_ROOT}/{station}/{year}/*.dat"))
    if not files:
        print(f"[{station}] {year}: 无 .dat 文件", flush=True)
        return
    with Pool(workers) as pool:
        results = pool.map(process_day, [(station, year, f) for f in files])
    results = [r for r in results if r is not None]
    if not results:
        print(f"[{station}] {year}: 无有效数据", flush=True)
        return
    df = pd.concat(results).sort_index()
    df = df[~df.index.duplicated()]

    omni_files = sorted(glob.glob(f"{OMNI_DIR}/omni_{year}*.csv"))
    omni = pd.concat([pd.read_csv(f, index_col=0, parse_dates=[0]) for f in omni_files]).sort_index()
    omni = omni[~omni.index.duplicated()]
    for c in OMNI_COLS:
        if c not in omni.columns:
            omni[c] = np.nan
    df = df.join(omni[OMNI_COLS], how="inner")
    df["station"] = station

    out = f"{OUT_DIR}/{station}_{year}_1min.csv.gz"
    df.to_csv(out, compression="gzip")
    n_days = df["coverage"].resample("D").first().notna().sum()
    print(f"[OK] {station} {year}: {len(df)} 分钟点, 有效天={n_days}/365, -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2015")
    ap.add_argument("--stations", default="HZT")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    for st in args.stations.split(","):
        run_station(st.strip(), args.year, args.workers)
    print("E0b DONE", flush=True)
