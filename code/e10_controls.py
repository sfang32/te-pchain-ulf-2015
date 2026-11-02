# -*- coding: utf-8 -*-
"""E10: 审稿回应控制分析（与 e3 主结果同口径：KSG l_HISTORY=1, 52x7d blocks,
shift surrogates 20x offsets 7-40d, k=4 neighbours, cadence 5-min）

子命令:
  chain   -- 13 站链平均 Pc5 指数作为 destination 的 TE-lag 扫描 (R1/E10a)
  lt      -- 台站地方时 day(06-18)/night(18-06) 分组的 20/45-min Pdyn excess TE (R4/E10c)
  var     -- Pdyn 变率(30-min rolling std) 作为源的 20/45-min TE, 3 代表站 (R4/E10d)
  ae      -- 高纬 3 站 AE 高/低三分位条件化的 20/45-min Pdyn excess TE (R4)
  ulfcmp  -- chain hourly 指数 vs Pilipenko ground ULF index 相关 (R1/E10b, 需外部数据)

输出: {ROOT}/jb_te_jastp/results/e10/*.csv
用法:
  export JAVA_HOME=/data/ssd-stor/fsf/softwares/java/jdk-20.0.2
  python e10_controls.py chain --workers 8
  python e10_controls.py lt   --workers 8
  python e10_controls.py var  --workers 8
  python e10_controls.py ae   --workers 8
"""
import argparse
import os
import numpy as np
import pandas as pd
import jpype

ROOT = "/data/ssd-stor/fsf/workspaces/research_ulf"
DATA_DIR = f"{ROOT}/jb_te_jastp/data"
OUT_DIR = f"{ROOT}/jb_te_jastp/results/e10"
JAR = "/data/ssd-stor/fsf/softwares/infodynamics-dist-1.5/infodynamics.jar"
PARAMS = ["Bz_GSM", "Flow_pressure", "Flow_speed", "Electric_field"]
PDYN = "Flow_pressure"
STATIONS13 = ["MHT", "MZL", "NAT", "SSL", "MLS", "PXT", "JFT",
              "HZT", "SYT", "ZQT", "SYS", "QZT", "LAT"]
# 地理经度 (Table 1, Phase-II 新码对应站)
LON = {"MHT": 122.3, "MZL": 117.4, "NAT": 125.0, "SSL": 116.2, "MLS": 118.5,
       "PXT": 103.9, "JFT": 114.5, "HZT": 120.2, "SYT": 111.5, "ZQT": 112.5,
       "SYS": 109.6, "QZT": 109.8, "LAT": 91.1}
CHUNK_DAYS = 7
MIN_CHUNK = 800
CADENCE = 5
N_SHIFT = 20


def init_jvm():
    if not jpype.isJVMStarted():
        jpype.startJVM(jpype.getDefaultJVMPath(), "-ea",
                       "-Djava.class.path=" + JAR)


def te_jidt(source, dest, knn=4):
    calc = jpype.JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov()
    calc.setProperty("k_HISTORY", "1")
    calc.setProperty("l_HISTORY", "1")
    calc.setProperty("k", str(knn))
    calc.setProperty("NOISE_LEVEL_TO_ADD", "1e-8")
    calc.initialise()
    calc.setObservations(np.asarray(source, float), np.asarray(dest, float))
    return float(calc.computeAverageLocalOfObservations())


def zscore(v):
    v = np.asarray(v, float)
    mu, sd = np.nanmean(v), np.nanstd(v)
    return (v - mu) / (sd if sd > 0 else 1.0)


def load_station(station, year="2015", cadence=CADENCE):
    df = pd.read_csv(f"{DATA_DIR}/{station}_{year}_1min.csv.gz",
                     index_col=0, parse_dates=[0])
    df = df.drop(columns=["station"], errors="ignore").select_dtypes(include=[np.number])
    if cadence > 1:
        df = df.resample(f"{cadence}min").mean()
    dest = pd.Series(zscore(np.log10(df["Pc5"].clip(lower=1e-6) + 1e-3)), index=df.index)
    srcs = {p: pd.Series(zscore(df[p]), index=df.index) for p in PARAMS if p in df.columns}
    return dest, srcs, df


def chunked_te(src, dst, n_chunks, min_chunk=MIN_CHUNK):
    vals = []
    edges = np.linspace(0, len(src), n_chunks + 1, dtype=int)
    for i in range(n_chunks):
        a, b = src[edges[i]:edges[i + 1]], dst[edges[i]:edges[i + 1]]
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() < min_chunk:
            continue
        vals.append(te_jidt(a[m], b[m]))
    return float(np.mean(vals)) if vals else np.nan


def excess_te(src, dst, lag_steps, n_chunks, seed=0):
    """true chunked TE + shift-surrogate null -> (te_true, null_mean, null_std, z, exceed)"""
    n = len(dst)
    lag = lag_steps
    te_true = chunked_te(src[: n - lag], dst[lag:], n_chunks)
    rng = np.random.default_rng(seed)
    lo, hi = 7 * 24 * 60 // CADENCE, 40 * 24 * 60 // CADENCE
    nulls = []
    for i in range(N_SHIFT):
        sh = int(rng.integers(lo, hi))
        nulls.append(chunked_te(np.roll(src, sh)[: n - lag], dst[lag:], n_chunks))
    nulls = np.array([x for x in nulls if not np.isnan(x)], float)
    mu, sd = (float(np.mean(nulls)), float(np.std(nulls))) if len(nulls) else (np.nan, np.nan)
    z = (te_true - mu) / sd if (sd and sd > 0) else np.nan
    exc = float(np.mean(nulls >= te_true)) if len(nulls) else np.nan
    return te_true, mu, sd, z, exc


def n_chunks_of(idx):
    return len(idx) // (CHUNK_DAYS * 24 * 60 // CADENCE)


# ---------------------------------------------------------------- chain (E10a)
def run_chain(args):
    dests, srcs_all = {}, None
    for st in STATIONS13:
        dest, srcs, _ = load_station(st)
        dests[st] = dest
        if srcs_all is None:
            srcs_all = srcs
    dd = pd.DataFrame(dests)
    chain = dd.mean(axis=1, skipna=True)          # 链平均 (zscore 后等权)
    chain = pd.Series(zscore(chain.values), index=chain.index)
    n_chunks = n_chunks_of(chain.index)
    rows = []
    for p, src in srcs_all.items():
        src = src.reindex(chain.index)
        for lag_min in range(5, 121, 5):
            lag = lag_min // CADENCE
            t, mu, sd, z, exc = excess_te(src.values, chain.values, lag, n_chunks,
                                          seed=7000 + lag_min)
            rows.append(dict(param=p, lag_min=lag_min, TE_true=t, null_mean=mu,
                             null_std=sd, z=z, exceed_pct=exc, n_stations=len(STATIONS13)))
            print(f"[chain {p}] lag={lag_min} z={z:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/e10a_chain_te.csv", index=False)
    print("E10a DONE -> e10a_chain_te.csv", flush=True)


# -------------------------------------------------------------------- lt (E10c)
def run_lt(args):
    lags = [20, 45]
    rows = []
    for st in STATIONS13:
        dest, srcs, _ = load_station(st)
        if PDYN not in srcs:
            continue
        src = srcs[PDYN]
        lt = (dest.index + pd.Timedelta(minutes=LON[st] / 15 * 60)).hour
        day_mask = (lt >= 6) & (lt < 18)
        n_chunks_full = n_chunks_of(dest.index)
        for grp, mask in (("day", day_mask), ("night", ~day_mask)):
            s = src[mask].values
            d = dest[mask].values
            nc = max(len(d) // (CHUNK_DAYS * 24 * 60 // CADENCE), 1)
            for lag_min in lags:
                t, mu, sd, z, exc = excess_te(s, d, lag_min // CADENCE, nc,
                                              seed=hash(st) % 10000 + lag_min)
                rows.append(dict(station=st, group=grp, lag_min=lag_min, TE_true=t,
                                 null_mean=mu, null_std=sd, z=z, exceed_pct=exc,
                                 n_samples=int(mask.sum())))
                print(f"[lt {st} {grp}] lag={lag_min} z={z:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/e10c_lt_split.csv", index=False)
    print("E10c DONE -> e10c_lt_split.csv", flush=True)


# ------------------------------------------------------- variability (E10d)
def run_var(args):
    reps = ["MHT", "PXT", "SYS"]          # OMOHE / OPIXI / OLEDO，与主文代表站一致
    rows = []
    for st in reps:
        dest, srcs, raw = load_station(st)
        if PDYN not in srcs:
            continue
        pdyn_5 = raw[PDYN].reindex(dest.index)
        var_src = pd.Series(zscore(pdyn_5.rolling(6).std().values), index=dest.index)
        n_chunks = n_chunks_of(dest.index)
        for lag_min in [20, 45]:
            t, mu, sd, z, exc = excess_te(var_src.values, dest.values, lag_min // CADENCE,
                                          n_chunks, seed=8100 + lag_min)
            rows.append(dict(station=st, source="Pdyn_std30", lag_min=lag_min, TE_true=t,
                             null_mean=mu, null_std=sd, z=z, exceed_pct=exc))
            print(f"[var {st}] lag={lag_min} z={z:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/e10d_variability.csv", index=False)
    print("E10d DONE -> e10d_variability.csv", flush=True)


# ------------------------------------------------------------------- ae (R4)
def run_ae(args):
    hl = ["MHT", "MZL", "NAT"]            # OMOHE / OMZLN / ONOAN
    ref, _, _ = load_station(hl[0])       # AE 来自 OMNI 合并列
    _, srcs_ref, raw_ref = load_station(hl[0])
    ae = raw_ref["AE-index"].reindex(ref.index)
    q1, q2 = ae.quantile([1 / 3, 2 / 3])
    hi_mask = ae >= q2
    lo_mask = ae <= q1
    rows = []
    for st in hl:
        dest, srcs, _ = load_station(st)
        if PDYN not in srcs:
            continue
        src = srcs[PDYN]
        for grp, mask in (("AE_high", hi_mask.values), ("AE_low", lo_mask.values)):
            s = src[mask].values
            d = dest[mask].values
            nc = max(len(d) // (CHUNK_DAYS * 24 * 60 // CADENCE), 1)
            for lag_min in [20, 45]:
                t, mu, sd, z, exc = excess_te(s, d, lag_min // CADENCE, nc,
                                              seed=9200 + lag_min)
                rows.append(dict(station=st, group=grp, lag_min=lag_min, TE_true=t,
                                 null_mean=mu, null_std=sd, z=z, exceed_pct=exc,
                                 n_samples=int(mask.sum())))
                print(f"[ae {st} {grp}] lag={lag_min} z={z:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/e10e_ae_conditioned.csv", index=False)
    print("E10e DONE -> e10e_ae_conditioned.csv", flush=True)


# ------------------------------------------------------- ULF index cmp (E10b)
def run_ulfcmp(args):
    """chain hourly index vs Pilipenko ground ULF index (hourly)."""
    ulf_path = args.ulf_file
    if not ulf_path or not os.path.exists(ulf_path):
        print("[SKIP] ULF index file not available:", ulf_path, flush=True)
        return
    dests = {}
    for st in STATIONS13:
        dest, _, _ = load_station(st)
        dests[st] = dest
    dd = pd.DataFrame(dests)
    chain_h = dd.mean(axis=1, skipna=True).resample("1h").mean()
    ulf = pd.read_csv(ulf_path)          # 列: time, Wpax(或 index)
    ulf["time"] = pd.to_datetime(ulf.iloc[:, 0])
    ulf = ulf.set_index("time")
    col = ulf.columns[-1]
    m = pd.concat([chain_h.rename("chain"), ulf[col].rename("wpax")], axis=1).dropna()
    from scipy.stats import spearmanr, pearsonr
    rs, ps = spearmanr(m["chain"], m["wpax"])
    rp, pp = pearsonr(m["chain"], m["wpax"])
    out = pd.DataFrame([dict(n=len(m), spearman_r=rs, spearman_p=ps,
                             pearson_r=rp, pearson_p=pp)])
    out.to_csv(f"{OUT_DIR}/e10b_ulfindex_corr.csv", index=False)
    print(f"E10b DONE: n={len(m)} spearman r={rs:.3f} (p={ps:.2g}) pearson r={rp:.3f}",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("chain")
    s1.add_argument("--workers", type=int, default=8)
    s2 = sub.add_parser("lt")
    s2.add_argument("--workers", type=int, default=8)
    s3 = sub.add_parser("var")
    s3.add_argument("--workers", type=int, default=8)
    s4 = sub.add_parser("ae")
    s4.add_argument("--workers", type=int, default=8)
    s5 = sub.add_parser("ulfcmp")
    s5.add_argument("--ulf-file", default="")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    init_jvm()
    dict(chain=run_chain, lt=run_lt, var=run_var, ae=run_ae, ulfcmp=run_ulfcmp)[args.cmd](args)
