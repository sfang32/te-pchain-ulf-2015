# -*- coding: utf-8 -*-
"""E3: TE 显著性检验 — 保自相关的时间平移（循环移位）surrogate
对每个 (station, param, lag):
  - true chunked TE（与 E2 相同：7-day 分块, Kraskov k=4, k/l_HISTORY=1）
  - null: n_shift 次整年循环移位源序列（7-40 天随机偏移，保留源/目标自相关，破坏耦合对齐）
  - 输出 TE_true, null_mean, null_std, z-score, exceed_pct
用法（按台站分片到不同服务器）:
  python e3_significance.py --stations MHT --n-shift 20 --workers 4
"""
import argparse
import os
import numpy as np
import pandas as pd
import jpype
from multiprocessing import Pool

ROOT = "/data/ssd-stor/fsf/workspaces/research_ulf"
DATA_DIR = f"{ROOT}/jb_te_jastp/data"
OUT_DIR = f"{ROOT}/jb_te_jastp/results/e3"
JAR = "/data/ssd-stor/fsf/softwares/infodynamics-dist-1.5/infodynamics.jar"
PARAMS = ["Bz_GSM", "Flow_pressure", "Flow_speed", "Electric_field"]
CHUNK_DAYS = 7
MIN_CHUNK = 800
CADENCE = 5


def init_jvm():
    if not jpype.isJVMStarted():
        jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + JAR)


def te_jidt(source, dest, knn=4):
    calc = jpype.JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov()
    calc.setProperty("k_HISTORY", "1")
    calc.setProperty("l_HISTORY", "1")
    calc.setProperty("k", str(knn))
    calc.setProperty("NOISE_LEVEL_TO_ADD", "1e-8")
    calc.initialise()
    calc.setObservations(np.asarray(source, float), np.asarray(dest, float))
    return float(calc.computeAverageLocalOfObservations())


def zscore(s: pd.Series):
    v = s.values.astype(float)
    mu, sd = np.nanmean(v), np.nanstd(v)
    return (v - mu) / (sd if sd > 0 else 1.0)


def load_station(station: str, year: str, cadence: int, band: str):
    df = pd.read_csv(f"{DATA_DIR}/{station}_{year}_1min.csv.gz", index_col=0, parse_dates=[0])
    df = df.drop(columns=["station"], errors="ignore").select_dtypes(include=[np.number])
    if cadence > 1:
        df = df.resample(f"{cadence}min").mean()
    dest = pd.Series(zscore(np.log10(df[band].clip(lower=1e-6) + 1e-3)), index=df.index)
    srcs = {p: pd.Series(zscore(df[p]), index=df.index) for p in PARAMS if p in df.columns}
    return dest, srcs


def chunked_te(src, dst, n_chunks):
    vals = []
    edges = np.linspace(0, len(src), n_chunks + 1, dtype=int)
    for i in range(n_chunks):
        a, b = src[edges[i]:edges[i + 1]], dst[edges[i]:edges[i + 1]]
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() < globals().get("MIN_CHUNK_EFF", MIN_CHUNK):
            continue
        vals.append(te_jidt(a[m], b[m]))
    if not vals:
        return np.nan
    return float(np.mean(vals))


def scan_one(args_tuple):
    station, year, band, cadence, lags, param, n_shift, seed_base = args_tuple
    try:
        init_jvm()
        dest, srcs = load_station(station, year, cadence, band)
    except FileNotFoundError:
        print(f"[SKIP] {station} 无数据", flush=True)
        return []
    if param not in srcs:
        return []
    src = srcs[param].values
    dst = dest.values
    n = len(dst)
    n_chunks = n // (CHUNK_DAYS * 24 * 60 // cadence)
    rng = np.random.default_rng(seed_base)
    shift_pool = rng.integers(7 * 24 * 60 // cadence, 40 * 24 * 60 // cadence,
                              size=n_shift * len(lags) + 10)
    rows = []
    k = 0
    for lag_min in lags:
        lag = lag_min // cadence
        if lag < 1:
            continue
        te_true = chunked_te(src[: n - lag], dst[lag:], n_chunks)
        nulls = []
        for i in range(n_shift):
            sh = int(shift_pool[k]); k += 1
            s_roll = np.roll(src, sh)
            nulls.append(chunked_te(s_roll[: n - lag], dst[lag:], n_chunks))
        nulls = np.array(nulls, dtype=float)
        nulls = nulls[~np.isnan(nulls)]
        mu, sd = float(np.mean(nulls)), float(np.std(nulls))
        rows.append(dict(station=station, param=param, lag_min=lag_min, band=band,
                         TE_true=te_true, null_mean=mu, null_std=sd,
                         z=(te_true - mu) / (sd if sd > 0 else np.nan),
                         exceed_pct=float(np.mean(nulls >= te_true)) if len(nulls) else np.nan,
                         n_null=len(nulls)))
        print(f"  [{station} {param}] lag={lag_min} z={rows[-1]['z']:.2f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2015")
    ap.add_argument("--stations", required=True)
    ap.add_argument("--band", default="Pc5")
    ap.add_argument("--lag-start", type=int, default=5)
    ap.add_argument("--lag-end", type=int, default=600)
    ap.add_argument("--lag-step", type=int, default=5)
    ap.add_argument("--cadence", type=int, default=5)
    ap.add_argument("--n-shift", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-chunk", type=int, default=800)
    args = ap.parse_args()
    globals()["MIN_CHUNK_EFF"] = args.min_chunk

    os.makedirs(OUT_DIR, exist_ok=True)
    lags = [m for m in range(args.lag_start, args.lag_end + 1, args.lag_step) if m % args.cadence == 0]
    tasks = []
    for si, st in enumerate(args.stations.split(",")):
        st = st.strip()
        for pi, p in enumerate(PARAMS):
            seed = 1000 * si + 10 * pi
            tasks.append((st, args.year, args.band, args.cadence, lags, p, args.n_shift, seed))

    with Pool(args.workers) as pool:
        all_rows = pool.map(scan_one, tasks)
    df = pd.DataFrame([r for rows in all_rows for r in rows])
    out = f"{OUT_DIR}/sig_{args.band}_{args.cadence}min_{'-'.join(args.stations.split(','))}.csv"
    df.to_csv(out, index=False)
    print(f"E3 DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
