# -*- coding: utf-8 -*-
"""E6a: TE 超参数审计 — 目的地历史长度 l_HISTORY 扫描（含循环移位 null）
目的：检验 E2/E3 的 TE 是否因 dest 历史条件化不足而被高估；
     为论文 Methods 提供超参数选择的依据图。
扫描: l_HISTORY ∈ {1,3,6,12,24} (5-min 步长 -> dest 记忆 5min~2h), k_HISTORY=1, Kraskov k=4
对象: MHT/SYS/PXT × {Flow_pressure, Bz_GSM, Flow_speed} × 代表性 lag 集
输出: results/e6a/hist_{station}.csv
"""
import argparse
import os
import numpy as np
import pandas as pd
import jpype
from multiprocessing import Pool

ROOT = "/data/ssd-stor/fsf/workspaces/research_ulf"
DATA_DIR = f"{ROOT}/jb_te_jastp/data"
OUT_DIR = f"{ROOT}/jb_te_jastp/results/e6a"
JAR = "/data/ssd-stor/fsf/softwares/infodynamics-dist-1.5/infodynamics.jar"
CHUNK_DAYS = 7
MIN_CHUNK = 800
CADENCE = 5
DEFAULT_LAGS = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 600]
L_HISTORIES = [1, 3, 6, 12, 24]


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


def zscore(s: pd.Series):
    v = s.values.astype(float)
    mu, sd = np.nanmean(v), np.nanstd(v)
    return (v - mu) / (sd if sd > 0 else 1.0)


def load_station(station, year, cadence, band="Pc5"):
    df = pd.read_csv(f"{DATA_DIR}/{station}_{year}_1min.csv.gz", index_col=0, parse_dates=[0])
    df = df.drop(columns=["station"], errors="ignore").select_dtypes(include=[np.number])
    if cadence > 1:
        df = df.resample(f"{cadence}min").mean()
    dest = pd.Series(zscore(np.log10(df[band].clip(lower=1e-6) + 1e-3)), index=df.index)
    srcs = {p: pd.Series(zscore(df[p]), index=df.index) for p in ["Bz_GSM", "Flow_pressure", "Flow_speed"]}
    return dest, srcs


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


def scan_one(args_tuple):
    station, year, param, n_shift, seed = args_tuple
    try:
        init_jvm()
        dest, srcs = load_station(station, year, CADENCE)
    except FileNotFoundError:
        print(f"[SKIP] {station} 无数据", flush=True)
        return []
    src = srcs[param].values
    dst = dest.values
    n = len(dst)
    n_chunks = n // (CHUNK_DAYS * 24 * 60 // CADENCE)
    rng = np.random.default_rng(seed)
    shift_pool = rng.integers(7 * 24 * 60 // CADENCE, 40 * 24 * 60 // CADENCE,
                              size=n_shift * len(DEFAULT_LAGS) * len(L_HISTORIES) + 10)
    rows = []
    k = 0
    for l_hist in L_HISTORIES:
        for lag_min in DEFAULT_LAGS:
            lag = lag_min // CADENCE
            if lag < l_hist:
                continue
            te_true = chunked_te(src[: n - lag], dst[lag:], n_chunks, l_hist)
            nulls = []
            for _ in range(n_shift):
                sh = int(shift_pool[k]); k += 1
                s_roll = np.roll(src, sh)
                nulls.append(chunked_te(s_roll[: n - lag], dst[lag:], n_chunks, l_hist))
            nulls = np.array(nulls, float); nulls = nulls[~np.isnan(nulls)]
            mu, sd = float(np.mean(nulls)), float(np.std(nulls))
            rows.append(dict(station=station, param=param, l_history=l_hist, lag_min=lag_min,
                             TE_true=te_true, null_mean=mu, null_std=sd,
                             z=(te_true - mu) / (sd if sd > 0 else np.nan),
                             excess=te_true - mu))
        print(f"  [{station} {param}] l_HISTORY={l_hist} done", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2015")
    ap.add_argument("--stations", required=True)
    ap.add_argument("--n-shift", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = []
    for si, st in enumerate(args.stations.split(",")):
        for pi, p in enumerate(["Flow_pressure", "Bz_GSM", "Flow_speed"]):
            tasks.append((st.strip(), args.year, p, args.n_shift, 7000 + 100 * si + pi))

    with Pool(args.workers) as pool:
        all_rows = pool.map(scan_one, tasks)
    df = pd.DataFrame([r for rows in all_rows for r in rows])
    out = f"{OUT_DIR}/hist_{'-'.join(args.stations.split(','))}.csv"
    df.to_csv(out, index=False)
    print(f"E6a DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
