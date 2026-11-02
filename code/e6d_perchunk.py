# -*- coding: utf-8 -*-
"""E6d-1: 逐 chunk TE 落盘（JIDT）— 供 block bootstrap 与置信区间
配置: 5-min cadence, k_HISTORY=1, l_HISTORY=6, k=4, 7-day 分块, lags {20,60,120} min
每 (station,param,lag): 52 个 chunk 的 TE_true 数组 + 20 次整年循环移位 null 的 chunk 均值数组
输出: results/e6d/perchunk_{station}.csv
"""
import argparse
import os
import numpy as np
import pandas as pd
import jpype
from multiprocessing import Pool

ROOT = "/data/ssd-stor/fsf/workspaces/research_ulf"
DATA_DIR = f"{ROOT}/jb_te_jastp/data"
OUT_DIR = f"{ROOT}/jb_te_jastp/results/e6d"
JAR = "/data/ssd-stor/fsf/softwares/infodynamics-dist-1.5/infodynamics.jar"
LAGS_MIN = [20, 60, 120]
N_SHIFT = 20
CHUNK_N = 7 * 24 * 12
MIN_CHUNK = 500
CAD = 5
L_HIST = 6


def init_jvm():
    if not jpype.isJVMStarted():
        jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + JAR)


def te_jidt(source, dest):
    calc = jpype.JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov()
    calc.setProperty("k_HISTORY", "1"); calc.setProperty("l_HISTORY", str(L_HIST))
    calc.setProperty("k", "4"); calc.setProperty("NOISE_LEVEL_TO_ADD", "1e-8")
    calc.initialise()
    calc.setObservations(np.asarray(source, float), np.asarray(dest, float))
    return float(calc.computeAverageLocalOfObservations())


def zscore(v):
    mu, sd = np.nanmean(v), np.nanstd(v)
    return (v - mu) / (sd if sd > 0 else 1.0)


def load_series(station, year, param):
    df = pd.read_csv(f"{DATA_DIR}/{station}_{year}_1min.csv.gz", index_col=0, parse_dates=[0])
    df = df.drop(columns=["station"], errors="ignore").select_dtypes(include=[np.number])
    df = df.resample(f"{CAD}min").mean()
    dest = zscore(np.log10(df["Pc5"].clip(lower=1e-6) + 1e-3).values)
    return zscore(df[param].values), dest


def chunk_edges(n, n_chunks):
    return np.linspace(0, n, n_chunks + 1, dtype=int)


def chunk_tes(src, dst, edges):
    vals = []
    for i in range(len(edges) - 1):
        a, b = src[edges[i]:edges[i + 1]], dst[edges[i]:edges[i + 1]]
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() < MIN_CHUNK:
            continue
        vals.append(te_jidt(a[m], b[m]))
    return vals


def scan_one(args_tuple):
    station, year, param, seed = args_tuple
    try:
        init_jvm()
        src, dst = load_series(station, year, param)
    except FileNotFoundError:
        return []
    n = len(dst)
    n_chunks = n // CHUNK_N
    edges = chunk_edges(n, n_chunks)
    rng = np.random.default_rng(seed)
    shifts = rng.integers(7 * 24 * 12, 41 * 24 * 12, size=N_SHIFT * len(LAGS_MIN) + 10)
    rows = []
    k = 0
    for lag_min in LAGS_MIN:
        lag = lag_min // CAD
        s_l, d_l = src[: n - lag], dst[lag:]
        tes = chunk_tes(s_l, d_l, edges)
        null_means = []
        for _ in range(N_SHIFT):
            s_r = np.roll(src, int(shifts[k])); k += 1
            nm = chunk_tes(s_r[: n - lag], d_l, edges)
            null_means.append(float(np.mean(nm)) if nm else np.nan)
        for ci, tv in enumerate(tes):
            rows.append(dict(station=station, param=param, lag_min=lag_min, chunk_id=ci, TE_chunk=tv,
                             null_mean=fnm if (fnm := float(np.mean(null_means))) else np.nan,
                             null_std=float(np.std(null_means))))
        print(f"  [{station} {param} lag{lag_min}] chunks={len(tes)}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2015")
    ap.add_argument("--stations", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = []
    for si, st in enumerate(args.stations.split(",")):
        for pi, p in enumerate(["Flow_pressure", "Bz_GSM"]):
            tasks.append((st.strip(), args.year, p, 9500 + 100 * si + pi))
    with Pool(args.workers) as pool:
        all_rows = pool.map(scan_one, tasks)
    df = pd.DataFrame([r for rows in all_rows for r in rows])
    out = f"{OUT_DIR}/perchunk_{'-'.join(args.stations.split(','))}.csv"
    df.to_csv(out, index=False)
    print(f"E6d DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
