# -*- coding: utf-8 -*-
"""E9: 天基-地面级联 TE — RBSP-B Pc5 功率 <-> 地面台站 Pc5 功率
计算: TE(RBSP->地面) 与对照 TE(地面->RBSP), lags {10..240} min
配置: 5-min cadence, k_HISTORY=1, l_HISTORY=6, k=4, 7-day 分块, 移位 null n_shift=12
输出: results/e9/cascade_{station}.csv
"""
import argparse
import os
import numpy as np
import pandas as pd
import jpype
from multiprocessing import Pool

ROOT = "/data/ssd-stor/fsf/workspaces/research_ulf"
DATA_DIR = f"{ROOT}/jb_te_jastp/data"
OUT_DIR = f"{ROOT}/jb_te_jastp/results/e9"
JAR = "/data/ssd-stor/fsf/softwares/infodynamics-dist-1.5/infodynamics.jar"
LAGS_MIN = [10, 20, 30, 45, 60, 90, 120, 180, 240]
N_SHIFT = 12
CHUNK_N = 7 * 24 * 12
MIN_CHUNK = 400
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


def load_pc5(station_or_rbsp, year="2015"):
    df = pd.read_csv(f"{DATA_DIR}/{station_or_rbsp}_{year}_1min.csv.gz", index_col=0, parse_dates=[0])
    df = df.drop(columns=["station"], errors="ignore").select_dtypes(include=[np.number])
    df = df.resample(f"{CAD}min").mean()
    return pd.Series(zscore(np.log10(df["Pc5"].clip(lower=1e-6) + 1e-3).values), index=df.index)


def chunked(src, dst, n_chunks):
    vals = []
    edges = np.linspace(0, len(src), n_chunks + 1, dtype=int)
    for i in range(n_chunks):
        a, b = src[edges[i]:edges[i + 1]], dst[edges[i]:edges[i + 1]]
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() < MIN_CHUNK:
            continue
        vals.append(te_jidt(a[m], b[m]))
    return vals


def scan(args_tuple):
    station, year, seed = args_tuple
    init_jvm()
    rbsp = load_pc5("RBSPB", year)
    gnd = load_pc5(station, year)
    idx = rbsp.index.intersection(gnd.index)
    rbsp, gnd = rbsp.loc[idx].values, gnd.loc[idx].values
    n = len(idx)
    n_chunks = n // CHUNK_N
    rng = np.random.default_rng(seed)
    shifts = rng.integers(7 * 24 * 12, 41 * 24 * 12, size=N_SHIFT * len(LAGS_MIN) * 2 + 10)
    rows = []
    k = 0
    for direction, s_all, d_all in [("RBSP_to_ground", rbsp, gnd), ("ground_to_RBSP", gnd, rbsp)]:
        for lag_min in LAGS_MIN:
            lag = lag_min // CAD
            s_l, d_l = s_all[: n - lag], d_all[lag:]
            tes = chunked(s_l, d_l, n_chunks)
            nulls = []
            for _ in range(N_SHIFT):
                s_r = np.roll(s_all, int(shifts[k])); k += 1
                nm = chunked(s_r[: n - lag], d_l, n_chunks)
                nulls.append(float(np.mean(nm)) if nm else np.nan)
            nulls = np.array(nulls, float); nulls = nulls[~np.isnan(nulls)]
            mu, sd = float(np.mean(nulls)), float(np.std(nulls))
            rows.append(dict(station=station, direction=direction, lag_min=lag_min,
                             TE=float(np.mean(tes)) if tes else np.nan, n_chunks=len(tes),
                             null_mean=mu, null_std=sd,
                             excess=(np.mean(tes) - mu) if tes else np.nan,
                             z=(np.mean(tes) - mu) / sd if sd > 0 else np.nan))
        print(f"  [{station} {direction}] done", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", default="MHT,PXT,SYS")
    ap.add_argument("--year", default="2015")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = [(st.strip(), args.year, 9900 + i) for i, st in enumerate(args.stations.split(","))]
    with Pool(args.workers) as pool:
        all_rows = pool.map(scan, tasks)
    df = pd.DataFrame([r for rows in all_rows for r in rows])
    out = f"{OUT_DIR}/cascade_{'-'.join(args.stations.split(','))}.csv"
    df.to_csv(out, index=False)
    print(f"E9 DONE -> {out}", flush=True)
