# -*- coding: utf-8 -*-
"""E6d-2: infomeasure 交叉验证（ksg 估计器, dest_hist_len=6 对齐 JIDT l_HISTORY=6）
在 infomeasure conda 环境运行（无 jpype 依赖）。
输出: results/e6d/crossval_infomeasure.csv（逐 chunk TE）
"""
import os
import numpy as np
import pandas as pd
import infomeasure as im
from multiprocessing import Pool

ROOT = "/data/ssd-stor/fsf/workspaces/research_ulf"
DATA_DIR = f"{ROOT}/jb_te_jastp/data"
OUT_DIR = f"{ROOT}/jb_te_jastp/results/e6d"
LAGS_MIN = [20, 60]
CHUNK_N = 7 * 24 * 12
MIN_CHUNK = 500
CAD = 5


def zscore(v):
    mu, sd = np.nanmean(v), np.nanstd(v)
    return (v - mu) / (sd if sd > 0 else 1.0)


def load_series(station, year, param):
    df = pd.read_csv(f"{DATA_DIR}/{station}_{year}_1min.csv.gz", index_col=0, parse_dates=[0])
    df = df.drop(columns=["station"], errors="ignore").select_dtypes(include=[np.number])
    df = df.resample(f"{CAD}min").mean()
    dest = zscore(np.log10(df["Pc5"].clip(lower=1e-6) + 1e-3).values)
    return zscore(df[param].values), dest


def scan_one(args_tuple):
    station, year, param = args_tuple
    try:
        src, dst = load_series(station, year, param)
    except FileNotFoundError:
        return []
    n = len(dst)
    edges = np.linspace(0, n, n // CHUNK_N + 1, dtype=int)
    rows = []
    for lag_min in LAGS_MIN:
        lag = lag_min // CAD
        s_l, d_l = src[: n - lag], dst[lag:]
        for ci in range(len(edges) - 1):
            a, b = s_l[edges[ci]:edges[ci + 1]], d_l[edges[ci]:edges[ci + 1]]
            m = ~(np.isnan(a) | np.isnan(b))
            if m.sum() < MIN_CHUNK:
                continue
            te = float(im.transfer_entropy(a[m].astype(float), b[m].astype(float),
                                           approach="ksg", k=4,
                                           src_hist_len=1, dest_hist_len=6, prop_time=1))
            rows.append(dict(station=station, param=param, lag_min=lag_min, chunk_id=ci, TE_im=te))
    print(f"  [{station} {param}] done", flush=True)
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", default="MHT,PXT")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = [(st.strip(), "2015", p) for st in args.stations.split(",") for p in ["Flow_pressure", "Bz_GSM"]]
    with Pool(4) as pool:
        all_rows = pool.map(scan_one, tasks)
    df = pd.DataFrame([r for rows in all_rows for r in rows])
    out = f"{OUT_DIR}/crossval_infomeasure.csv"
    df.to_csv(out, index=False)
    print(f"E6d-infomeasure DONE -> {out}", flush=True)
