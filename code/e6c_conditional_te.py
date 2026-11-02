# -*- coding: utf-8 -*-
"""E6c: 条件转移熵 — 分离动压/重联通道的独立贡献
配置: 5-min cadence, l_HISTORY=6, Kraskov k=4, 7-day 分块, lag ∈ {20,60,120,240,360,600} min
计算:
  TE(Pdyn→Pc5)         vs  TE(Pdyn→Pc5 | Bz)
  TE(Bz→Pc5)           vs  TE(Bz→Pc5 | Pdyn)
null: 源与条件变量用同一随机偏移整年循环移位（保留 Pdyn-Bz 互耦合，只破坏与目标对齐）, n_shift=10
输出: results/e6c/cond_{station}.csv
"""
import argparse
import os
import numpy as np
import pandas as pd
import jpype
from jpype import JArray, JDouble, JObject
from multiprocessing import Pool

_JD = None  # 惰性初始化（需 JVM 已启动）


def jd_types():
    global _JD, _JD2
    if _JD is None:
        _JD = JArray(JDouble)
        _JD2 = JArray(_JD)
    return _JD, _JD2

ROOT = "/data/ssd-stor/fsf/workspaces/research_ulf"
DATA_DIR = f"{ROOT}/jb_te_jastp/data"
OUT_DIR = f"{ROOT}/jb_te_jastp/results/e6c"
JAR = "/data/ssd-stor/fsf/softwares/infodynamics-dist-1.5/infodynamics.jar"
LAGS_MIN = [20, 60, 120, 240, 360, 600]
N_SHIFT = 10
CHUNK_N = 7 * 24 * 12  # 7-day @5min
MIN_CHUNK = 500
CAD = 5
L_HIST = 6


def init_jvm():
    if not jpype.isJVMStarted():
        jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + JAR)


def zscore(v):
    mu, sd = np.nanmean(v), np.nanstd(v)
    return (v - mu) / (sd if sd > 0 else 1.0)


def load_pair(station, year):
    df = pd.read_csv(f"{DATA_DIR}/{station}_{year}_1min.csv.gz", index_col=0, parse_dates=[0])
    df = df.drop(columns=["station"], errors="ignore").select_dtypes(include=[np.number])
    df = df.resample(f"{CAD}min").mean()
    dest = zscore(np.log10(df["Pc5"].clip(lower=1e-6) + 1e-3).values)
    return {"Flow_pressure": zscore(df["Flow_pressure"].values),
            "Bz_GSM": zscore(df["Bz_GSM"].values)}, dest, df.index


def cond_te(src, dst, cond, n_chunks, lag):
    """条件 TE: src(t) -> dst(t+lag) | cond(t), 均手动对齐后调用 JIDT(DELAY=1 内部)"""
    jd1, jd2 = jd_types()
    vals = []
    edges = np.linspace(0, len(dst) - lag, n_chunks + 1, dtype=int)
    for i in range(n_chunks):
        a, b, e = src[edges[i]:edges[i + 1]], dst[lag + edges[i]:lag + edges[i + 1]], cond[edges[i]:edges[i + 1]]
        m = ~(np.isnan(a) | np.isnan(b) | np.isnan(e))
        if m.sum() < MIN_CHUNK:
            continue
        calc = jpype.JPackage("infodynamics.measures.continuous.kraskov").ConditionalTransferEntropyCalculatorKraskov()
        calc.setProperty("k_HISTORY", "1")
        calc.setProperty("l_HISTORY", str(L_HIST))
        calc.setProperty("k", "4")
        calc.setProperty("COND_EMBED_LENGTHS", "1")
        calc.setProperty("NOISE_LEVEL_TO_ADD", "1e-8")
        calc.initialise()
        calc.setObservations(JObject(a[m].astype(float), jd1),
                             JObject(b[m].astype(float), jd1),
                             JObject(e[m].reshape(-1, 1).astype(float), jd2))
        vals.append(float(calc.computeAverageLocalOfObservations()))
    return float(np.mean(vals)) if vals else np.nan


def uncond_te(src, dst, n_chunks, lag):
    vals = []
    edges = np.linspace(0, len(dst) - lag, n_chunks + 1, dtype=int)
    for i in range(n_chunks):
        a, b = src[edges[i]:edges[i + 1]], dst[lag + edges[i]:lag + edges[i + 1]]
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() < MIN_CHUNK:
            continue
        calc = jpype.JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov()
        calc.setProperty("k_HISTORY", "1")
        calc.setProperty("l_HISTORY", str(L_HIST))
        calc.setProperty("k", "4")
        calc.setProperty("NOISE_LEVEL_TO_ADD", "1e-8")
        calc.initialise()
        calc.setObservations(a[m].astype(float), b[m].astype(float))
        vals.append(float(calc.computeAverageLocalOfObservations()))
    return float(np.mean(vals)) if vals else np.nan


def scan_channel(args_tuple):
    station, year, src_name, cond_name, seed = args_tuple
    try:
        init_jvm()
        srcs, dst, idx = load_pair(station, year)
    except FileNotFoundError:
        print(f"[SKIP] {station}", flush=True)
        return []
    n = len(dst)
    n_chunks = n // CHUNK_N
    rng = np.random.default_rng(seed)
    shifts = rng.integers(7 * 24 * 12, 41 * 24 * 12, size=N_SHIFT * len(LAGS_MIN) * 2 + 10)
    rows = []
    k = 0
    for lag_min in LAGS_MIN:
        lag = lag_min // CAD
        src, cond = srcs[src_name], srcs[cond_name]
        te_u = uncond_te(src[: n - lag], dst, n_chunks, lag)
        te_c = cond_te(src[: n - lag], dst, cond[: n - lag], n_chunks, lag)
        null_u, null_c = [], []
        for _ in range(N_SHIFT):
            sh = int(shifts[k]); k += 1
            s_r, c_r = np.roll(src, sh), np.roll(cond, sh)
            null_u.append(uncond_te(s_r[: n - lag], dst, n_chunks, lag))
            null_c.append(cond_te(s_r[: n - lag], dst, c_r[: n - lag], n_chunks, lag))
        nu, nc = np.array(null_u, float), np.array(null_c, float)
        rows.append(dict(station=station, source=src_name, conditional=cond_name, lag_min=lag_min,
                         TE_unc=te_u, null_unc_mean=float(np.nanmean(nu)),
                         z_unc=(te_u - np.nanmean(nu)) / (np.nanstd(nu) if np.nanstd(nu) > 0 else np.nan),
                         TE_cond=te_c, null_cond_mean=float(np.nanmean(nc)),
                         z_cond=(te_c - np.nanmean(nc)) / (np.nanstd(nc) if np.nanstd(nc) > 0 else np.nan),
                         excess_unc=te_u - np.nanmean(nu), excess_cond=te_c - np.nanmean(nc)))
        print(f"  [{station} {src_name}|{cond_name}] lag={lag_min} TE_u={te_u:.4f} TE_c={te_c:.4f}", flush=True)
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
        st = st.strip()
        tasks.append((st, args.year, "Flow_pressure", "Bz_GSM", 8100 + si))
        tasks.append((st, args.year, "Bz_GSM", "Flow_pressure", 8200 + si))
    with Pool(args.workers) as pool:
        all_rows = pool.map(scan_channel, tasks)
    df = pd.DataFrame([r for rows in all_rows for r in rows])
    out = f"{OUT_DIR}/cond_{'-'.join(args.stations.split(','))}.csv"
    df.to_csv(out, index=False)
    print(f"E6c DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
