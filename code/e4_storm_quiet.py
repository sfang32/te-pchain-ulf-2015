# -*- coding: utf-8 -*-
"""E4: 磁暴/宁静分层的转移熵
事件定义（SYM-H 来自站文件内的 OMNI 列）:
  storm: SYM-H <= -50 nT 的 episode（间隔<6h 合并），onset 起 48h 窗
  quiet: SYM-H > -5 nT 连续 >=24h 的时段，取 24h 窗（数量上限 40）
逐 chunk 计算（5-min cadence, l_HISTORY=6, k=4）:
  TE(Pdyn->Pc5) 与 TE(Bz->Pc5 | Pdyn)，lags {10,20,30,45,60,90,120} min
null: chunk 内循环移位（偏移 >=1/3 长度），n_shift=20
输出: results/e4/stq_{station}.csv
"""
import argparse
import os
import numpy as np
import pandas as pd
import jpype
from jpype import JArray, JDouble, JObject
from multiprocessing import Pool

ROOT = "/data/ssd-stor/fsf/workspaces/research_ulf"
DATA_DIR = f"{ROOT}/jb_te_jastp/data"
OUT_DIR = f"{ROOT}/jb_te_jastp/results/e4"
LAGS_MIN = [10, 20, 30, 45, 60, 90, 120]
CAD = 5
L_HIST = 6
N_SHIFT = 20
STORM_THR = -50.0
QUIET_THR = -5.0
STORM_SPAN_H = 48
MAX_QUIET = 40


def init_jvm():
    global _JD1, _JD2
    if not jpype.isJVMStarted():
        jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" +
                       "/data/ssd-stor/fsf/softwares/infodynamics-dist-1.5/infodynamics.jar")
    _JD1 = JArray(JDouble)
    _JD2 = JArray(_JD1)


def te_unc(a, b, l_hist):
    calc = jpype.JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov()
    calc.setProperty("k_HISTORY", "1"); calc.setProperty("l_HISTORY", str(l_hist))
    calc.setProperty("k", "4"); calc.setProperty("NOISE_LEVEL_TO_ADD", "1e-8")
    calc.initialise()
    calc.setObservations(JObject(a.astype(float), _JD1), JObject(b.astype(float), _JD1))
    return float(calc.computeAverageLocalOfObservations())


def te_cond(a, b, e, l_hist):
    calc = jpype.JPackage("infodynamics.measures.continuous.kraskov").ConditionalTransferEntropyCalculatorKraskov()
    calc.setProperty("k_HISTORY", "1"); calc.setProperty("l_HISTORY", str(l_hist))
    calc.setProperty("k", "4"); calc.setProperty("COND_EMBED_LENGTHS", "1")
    calc.setProperty("NOISE_LEVEL_TO_ADD", "1e-8")
    calc.initialise()
    calc.setObservations(JObject(a.astype(float), _JD1), JObject(b.astype(float), _JD1),
                         JObject(e.reshape(-1, 1).astype(float), _JD2))
    return float(calc.computeAverageLocalOfObservations())


def zscore(v):
    mu, sd = np.nanmean(v), np.nanstd(v)
    return (v - mu) / (sd if sd > 0 else 1.0)


def build_chunks(symh, idx):
    """返回 [(start_i, end_i, type, label)]，索引为 5-min 序列下标"""
    s = pd.Series(symh.values, index=idx)
    bad = s.isna().values
    chunks = []
    # storm episodes
    below = (s <= STORM_THR).values
    i = 0
    n = len(below)
    onsets = []
    while i < n:
        if below[i] and not bad[i]:
            j = i
            gap = 0
            while j < n and gap < 6 * 12:  # 6h 合并窗
                if below[j] and not bad[j]:
                    gap = 0
                else:
                    gap += 1
                j += 1
            onsets.append(i)
            i = j + 6 * 12
        else:
            i += 1
    for oi in onsets:
        ei = min(oi + STORM_SPAN_H * 12, n - L_HIST - max(LAGS_MIN) // CAD - 2)
        if ei - oi > 12 * 12:  # 至少 12h
            chunks.append((oi, ei, "storm", str(idx[oi].date())))
    # quiet stretches: SYM-H > -5 连续 24h
    above = (s > QUIET_THR).values
    i = 0
    qchunks = []
    while i < n and len(qchunks) < MAX_QUIET:
        if above[i] and not bad[i]:
            j = i
            while j < n and above[j] and not bad[j]:
                j += 1
            span = j - i
            k = i
            while span >= 24 * 12 and len(qchunks) < MAX_QUIET:
                qs, qe = k, k + 24 * 12
                if qe < n:
                    # 与已有 storm chunk 重叠则跳过
                    if not any(not (qe < so or qs > se) for so, se, _, _ in chunks):
                        qchunks.append((qs, qe, "quiet", str(idx[qs].date())))
                        k = qe + 12  # 间隔 1h
                        span = j - k
                    else:
                        k += 12 * 12
                        span = j - k
                else:
                    break
            i = j
        else:
            i += 1
    return chunks + qchunks


def process_chunk(args):
    station, year, ci, cinfo = args
    init_jvm()
    df = pd.read_csv(f"{DATA_DIR}/{station}_{year}_1min.csv.gz", index_col=0, parse_dates=[0])
    df = df.drop(columns=["station"], errors="ignore").select_dtypes(include=[np.number])
    df = df.resample(f"{CAD}min").mean()
    symh = df["SYM-H"].values
    dest_all = zscore(np.log10(df["Pc5"].clip(lower=1e-6) + 1e-3).values)
    pdyn = zscore(df["Flow_pressure"].values)
    bz = zscore(df["Bz_GSM"].values)
    rng = np.random.default_rng(9000 + ci)
    rows = []
    si, ei, ctype, label = cinfo
    for lag_min in LAGS_MIN:
        lag = lag_min // CAD
        a_p, a_b, b_, e_ = pdyn[si:ei - lag], bz[si:ei - lag], dest_all[si + lag:ei], pdyn[si + lag - lag:ei - lag]
        m = ~(np.isnan(a_p) | np.isnan(a_b) | np.isnan(b_) | np.isnan(e_))
        if m.sum() < 100:
            continue
        aP, aB, B, E = a_p[m], a_b[m], b_[m], e_[m]
        te_p = te_unc(aP, B, L_HIST)
        te_bz_c = te_cond(aB, B, E, L_HIST)
        nu_p, nu_c = [], []
        for _ in range(N_SHIFT):
            sh = rng.integers(m.sum() // 3, max(m.sum() // 3 + 1, 2 * m.sum() // 3))
            aP_r, aB_r, E_r = np.roll(aP, sh), np.roll(aB, sh), np.roll(E, sh)
            nu_p.append(te_unc(aP_r, B, L_HIST))
            nu_c.append(te_cond(aB_r, B, E_r, L_HIST))
        nu_p, nu_c = np.array(nu_p, float), np.array(nu_c, float)
        rows.append(dict(station=station, chunk_type=ctype, event=label, chunk_id=ci,
                         lag_min=lag_min, n=len(B), symh_min=float(np.nanmin(symh[si:ei])),
                         TE_Pdyn=te_p, null_Pdyn=float(np.mean(nu_p)),
                         z_Pdyn=(te_p - np.mean(nu_p)) / (np.std(nu_p) if np.std(nu_p) > 0 else np.nan),
                         TE_Bz_cond=te_bz_c, null_Bz_cond=float(np.mean(nu_c)),
                         z_Bz_cond=(te_bz_c - np.mean(nu_c)) / (np.std(nu_c) if np.std(nu_c) > 0 else np.nan)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2015")
    ap.add_argument("--stations", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    for st in args.stations.split(","):
        st = st.strip()
        df = pd.read_csv(f"{DATA_DIR}/{st}_{args.year}_1min.csv.gz", index_col=0, parse_dates=[0])
        df = df.drop(columns=["station"], errors="ignore").select_dtypes(include=[np.number])
        df = df.resample(f"{CAD}min").mean()
        chunks = build_chunks(df["SYM-H"], df.index)
        print(f"[{st}] chunks: {len(chunks)} (storm={sum(1 for c in chunks if c[2]=='storm')}, "
              f"quiet={sum(1 for c in chunks if c[2]=='quiet')})", flush=True)
        tasks = [(st, args.year, ci, cinfo) for ci, cinfo in enumerate(chunks)]
        with Pool(args.workers) as pool:
            all_rows = pool.map(process_chunk, tasks)
        out = pd.DataFrame([r for rows in all_rows for r in rows])
        path = f"{OUT_DIR}/stq_{st}.csv"
        out.to_csv(path, index=False)
        print(f"E4 {st} DONE -> {path}", flush=True)


if __name__ == "__main__":
    main()
