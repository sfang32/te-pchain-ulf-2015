# -*- coding: utf-8 -*-
"""E6b: surrogate 加严 + 小时级分辨率对照
变体（dest=Pc5 log功率, l_HISTORY: 5-min 用 6, 1-h 用 1 和 2）:
  A shift40  : 整年循环移位 7-40 天（=E3 原版, 基准）
  B shift300 : 移位 1-300 天（破 27 天太阳自转共模）
  C highpass : 两序列去 24h 滑动均值后再算（去慢趋势）, 移位 7-40 天
  D hourly   : 1-h 分辨率, 30-day 分块, lag 1-24 h, l_HISTORY=1,2, 移位 1-300 天
输出: results/e6b/robust_{station}.csv
"""
import argparse
import os
import numpy as np
import pandas as pd
import jpype
from multiprocessing import Pool

ROOT = "/data/ssd-stor/fsf/workspaces/research_ulf"
DATA_DIR = f"{ROOT}/jb_te_jastp/data"
OUT_DIR = f"{ROOT}/jb_te_jastp/results/e6b"
JAR = "/data/ssd-stor/fsf/softwares/infodynamics-dist-1.5/infodynamics.jar"
PARAMS = ["Flow_pressure", "Bz_GSM", "Flow_speed"]
LAGS_5MIN = [20, 60, 120, 240, 360, 600]
N_SHIFT = 12
MIN_CHUNK = 500


def init_jvm():
    if not jpype.isJVMStarted():
        jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + JAR)


def te_jidt(source, dest, l_hist, knn=4):
    calc = jpype.JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov()
    calc.setProperty("k_HISTORY", "1")
    calc.setProperty("l_HISTORY", str(l_hist))
    calc.setProperty("k", str(knn))
    calc.setProperty("NOISE_LEVEL_TO_ADD", "1e-8")
    calc.initialise()
    calc.setObservations(np.asarray(source, float), np.asarray(dest, float))
    return float(calc.computeAverageLocalOfObservations())


def zscore(v):
    mu, sd = np.nanmean(v), np.nanstd(v)
    return (v - mu) / (sd if sd > 0 else 1.0)


def load_series(station, year, cadence_min, param, band="Pc5"):
    df = pd.read_csv(f"{DATA_DIR}/{station}_{year}_1min.csv.gz", index_col=0, parse_dates=[0])
    df = df.drop(columns=["station"], errors="ignore").select_dtypes(include=[np.number])
    if cadence_min > 1:
        df = df.resample(f"{cadence_min}min").mean()
    dest = pd.Series(zscore(np.log10(df[band].clip(lower=1e-6) + 1e-3).values), index=df.index)
    src = pd.Series(zscore(df[param].values), index=df.index)
    return dest, src


def chunked_te(src, dst, n_chunks, l_hist):
    vals = []
    edges = np.linspace(0, len(src), n_chunks + 1, dtype=int)
    for i in range(n_chunks):
        a, b = src[edges[i]:edges[i + 1]], dst[edges[i]:edges[i + 1]]
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() < MIN_CHUNK:
            continue
        vals.append(te_jidt(a[m], b[m], l_hist))
    return float(np.mean(vals)) if vals else np.nan


def run_variant(station, year, variant, param):
    init_jvm()
    rng = np.random.default_rng(hash((station, variant, param)) % (2**31))
    rows = []
    if variant == "D_hourly":
        cad = 60
        dest, src = load_series(station, year, cad, param)
        n = len(dest)
        n_chunks = n // (30 * 24)  # 30-day hourly chunks
        lags_min = list(range(60, 1441, 60))
        shifts = rng.integers(24, 720, size=N_SHIFT * len(lags_min) * 2 + 10)  # 1-300 天(小时样本)
        k = 0
        for l_hist in (1, 2):
            for lag_min in lags_min:
                lag = lag_min // cad
                if lag < l_hist:
                    continue
                te = chunked_te(src.values[: n - lag], dest.values[lag:], n_chunks, l_hist)
                nulls = []
                for _ in range(N_SHIFT):
                    s_roll = np.roll(src.values, int(shifts[k])); k += 1
                    nulls.append(chunked_te(s_roll[: n - lag], dest.values[lag:], n_chunks, l_hist))
                nulls = np.array(nulls, float); nulls = nulls[~np.isnan(nulls)]
                rows.append(dict(variant=variant, param=param, l_history=l_hist, lag_min=lag_min,
                                 TE_true=te, null_mean=float(np.mean(nulls)),
                                 null_std=float(np.std(nulls)), n_shift=len(nulls),
                                 z=(te - np.mean(nulls)) / (np.std(nulls) if np.std(nulls) > 0 else np.nan)))
    else:
        cad = 5
        dest, src = load_series(station, year, cad, param)
        if variant == "C_highpass":
            win = 24 * 60 // cad
            dest = dest - dest.rolling(win, center=True, min_periods=win // 4).mean()
            src = src - src.rolling(win, center=True, min_periods=win // 4).mean()
        n = len(dest)
        n_chunks = n // (CHUNK := 7 * 24 * 60 // cad)
        l_hist = 6
        shifts = rng.integers(7 * 288 if False else 7 * 24 * 12, 41 * 24 * 12, size=N_SHIFT * len(LAGS_5MIN) + 10) \
            if variant in ("A_shift40", "C_highpass") else \
            rng.integers(1 * 24 * 12, 301 * 24 * 12, size=N_SHIFT * len(LAGS_5MIN) + 10)
        k = 0
        for lag_min in LAGS_5MIN:
            lag = lag_min // cad
            te = chunked_te(src.values[: n - lag], dest.values[lag:], n_chunks, l_hist)
            nulls = []
            for _ in range(N_SHIFT):
                s_roll = np.roll(src.values, int(shifts[k])); k += 1
                nulls.append(chunked_te(s_roll[: n - lag], dest.values[lag:], n_chunks, l_hist))
            nulls = np.array(nulls, float); nulls = nulls[~np.isnan(nulls)]
            rows.append(dict(variant=variant, param=param, l_history=l_hist, lag_min=lag_min,
                             TE_true=te, null_mean=float(np.mean(nulls)),
                             null_std=float(np.std(nulls)), n_shift=len(nulls),
                             z=(te - np.mean(nulls)) / (np.std(nulls) if np.std(nulls) > 0 else np.nan)))
    print(f"  [{station} {param} {variant}] done", flush=True)
    return rows


def scan_one(args_tuple):
    station, year, variant, param = args_tuple
    try:
        return run_variant(station, year, variant, param)
    except FileNotFoundError:
        print(f"[SKIP] {station} 无数据", flush=True)
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2015")
    ap.add_argument("--stations", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    variants = ["A_shift40", "B_shift300", "C_highpass", "D_hourly"]
    tasks = [(st.strip(), args.year, v, p) for st in args.stations.split(",")
             for v in variants for p in PARAMS]
    with Pool(args.workers) as pool:
        all_rows = pool.map(scan_one, tasks)
    df = pd.DataFrame([r for rows in all_rows for r in rows])
    out = f"{OUT_DIR}/robust_{'-'.join(args.stations.split(','))}.csv"
    df.to_csv(out, index=False)
    print(f"E6b DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
