# -*- coding: utf-8 -*-
"""E2: 主 TE 扫描 — 太阳风参数 -> 台站带限 ULF 波功率（log 功率）
按 lag 分片并行（跨服务器），每片输出一个 CSV。
设计：7-day 滑动块内计算 Kraskov TE（k_HISTORY=1, l_HISTORY=1, k=4），块均值±std + 有效块数。
默认正向 TE（4 个太阳风参数）；--reverse 1 时加算反向（零假设对照）。

用法（每台服务器一个 lag 块）:
  python e2_te_scan.py --lag-start 5 --lag-end 100 --lag-step 5 \
      --band Pc5 --cadence 5 --stations MHT,MZL,NAT,MLS,JFT,HZT,PXT,LAT,SYS --workers 4
"""
import argparse
import jpype
import numpy as np
import pandas as pd
from multiprocessing import Pool

ROOT = "/data/ssd-stor/fsf/workspaces/research_ulf"
DATA_DIR = f"{ROOT}/jb_te_jastp/data"
OUT_DIR = f"{ROOT}/jb_te_jastp/results/e2"
JAR = "/data/ssd-stor/fsf/softwares/infodynamics-dist-1.5/infodynamics.jar"
PARAMS = ["Bz_GSM", "Flow_pressure", "Flow_speed", "Electric_field"]
CHUNK_DAYS = 7
MIN_CHUNK = 800


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


def chunked_te(src: np.ndarray, dst: np.ndarray, n_chunks: int):
    """按时间块计算 TE, 返回 (mean, std, n_valid, total_n)"""
    vals, tot = [], 0
    edges = np.linspace(0, len(src), n_chunks + 1, dtype=int)
    for i in range(n_chunks):
        a, b = src[edges[i]:edges[i + 1]], dst[edges[i]:edges[i + 1]]
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() < MIN_CHUNK:
            continue
        vals.append(te_jidt(a[m], b[m]))
        tot += int(m.sum())
    if not vals:
        return np.nan, np.nan, 0, tot
    return float(np.mean(vals)), float(np.std(vals)), len(vals), tot


def load_station(station: str, year: str, cadence: int, band: str):
    df = pd.read_csv(f"{DATA_DIR}/{station}_{year}_1min.csv.gz", index_col=0, parse_dates=[0])
    df = df.drop(columns=["station"], errors="ignore")
    df = df.select_dtypes(include=[np.number])  # 去掉 object 列，防止 resample 报错
    if cadence > 1:
        df = df.resample(f"{cadence}min").mean()
    dest = pd.Series(zscore(np.log10(df[band].clip(lower=1e-6) + 1e-3)), index=df.index)
    srcs = {p: pd.Series(zscore(df[p]), index=df.index) for p in PARAMS if p in df.columns}
    return dest, srcs


def scan_params(args_tuple):
    """worker: 对单个参数做全 lag 扫描（每进程独立 JVM）。lags 为分钟列表"""
    station, year, band, cadence, lags, param, reverse = args_tuple
    try:
        init_jvm()
        dest, srcs = load_station(station, year, cadence, band)
    except FileNotFoundError:
        print(f"  [SKIP] {station} {year} 无数据文件", flush=True)
        return []
    if param not in srcs:
        print(f"  [SKIP] {station} 缺少参数 {param}", flush=True)
        return []
    src = srcs[param].values
    dst = dest.values
    n_chunks = len(dst) // (CHUNK_DAYS * 24 * 60 // cadence)
    rows = []
    for lag_min in lags:
        lag = lag_min // cadence  # 转为 cadence 步数
        if lag < 1:
            continue
        # 源超前 lag 个 cadence 步
        s_shift = src[: len(src) - lag]
        d_lag = dst[lag:]
        te_f, sd_f, n_f, tot_f = chunked_te(s_shift, d_lag, n_chunks)
        row = dict(station=station, param=param, lag_min=lag_min, band=band,
                   cadence_min=cadence, TE_fwd=te_f, TE_fwd_std=sd_f, n_chunks_fwd=n_f,
                   n_points_fwd=tot_f)
        if reverse:
            te_r, sd_r, n_r, tot_r = chunked_te(d_lag, s_shift, n_chunks)
            row.update(TE_rev=te_r, TE_rev_std=sd_r, n_chunks_rev=n_r)
        rows.append(row)
        print(f"  [{station} {param}] lag={lag_min}min TE_fwd={te_f:.4f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2015")
    ap.add_argument("--stations", default="MHT,MZL,NAT,MLS,JFT,HZT,PXT,LAT,SYS")
    ap.add_argument("--band", default="Pc5")
    ap.add_argument("--cadence", type=int, default=5)
    ap.add_argument("--lag-start", type=int, default=5)
    ap.add_argument("--lag-end", type=int, default=100)
    ap.add_argument("--lag-step", type=int, default=5)
    ap.add_argument("--reverse", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    lags = [m for m in range(args.lag_start, args.lag_end + 1, args.lag_step) if m % args.cadence == 0]
    tasks = []
    for st in args.stations.split(","):
        for p in PARAMS:
            tasks.append((st.strip(), args.year, args.band, args.cadence, lags, p, args.reverse))

    with Pool(args.workers) as pool:
        all_rows = pool.map(scan_params, tasks)
    df = pd.DataFrame([r for rows in all_rows for r in rows]).sort_values(["station", "param", "lag_min"])
    out = f"{OUT_DIR}/te_scan_{args.band}_{args.cadence}min_lag{args.lag_start}-{args.lag_end}{args.tag}.csv"
    df.to_csv(out, index=False)
    print(f"E2 BLOCK DONE ({args.lag_start}-{args.lag_end} min) -> {out}", flush=True)


if __name__ == "__main__":
    main()
