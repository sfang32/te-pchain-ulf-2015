# -*- coding: utf-8 -*-
"""E10b: chain-average hourly Pc5 index vs Pilipenko ground ULF wave index (2015, hourly).

Data: /data/ftp/fsf/databases/data_ulf/data_ulf_index/2015/ULF_15MM.asc
  columns: DD HH Tgr Tgeo Timf Tden V N Bz Dst  (99.99 / NaN = missing)
  Tgr/Tgeo: ground ULF wave power indices (geomagnetic pulsation bands)
Output: /tmp/e10/e10b_ulfindex_corr.csv + stdout summary
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

DATA_DIR = "/data/ssd-stor/fsf/workspaces/research_ulf/jb_te_jastp/data"
IDX_DIR = "/data/ftp/fsf/databases/data_ulf/data_ulf_index/2015"
OUT = "/tmp/e10/e10b_ulfindex_corr.csv"
STATIONS = ["MHT", "MZL", "NAT", "SSL", "MLS", "PXT", "JFT",
            "HZT", "SYT", "ZQT", "SYS", "QZT", "LAT"]

MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
          "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}


def parse_asc(path):
    """Parse one monthly ULF index .asc file into an hourly DataFrame."""
    rows = []
    month = None
    with open(path, "r", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0].upper() in MONTHS:          # header line: 'MAR 2015'
                month = MONTHS[parts[0].upper()]
                continue
            if len(parts) < 10 or month is None:    # column-name header
                continue
            try:
                dd, hh = int(parts[0]), int(parts[1])
                vals = [np.nan if p in ("NaN", "99.99", "99.99.") else float(p)
                        for p in parts[2:10]]
            except ValueError:
                continue
            rows.append((pd.Timestamp(2015, month, dd, hh), *vals))
    df = pd.DataFrame(rows, columns=["time", "Tgr", "Tgeo", "Timf", "Tden",
                                     "V", "N", "Bz", "Dst"]).set_index("time")
    return df


def main():
    # --- 1. Pilipenko hourly index, all 12 months of 2015 ---
    idx = pd.concat([parse_asc(p) for p in sorted(glob.glob(f"{IDX_DIR}/ULF_15*.asc"))])
    idx = idx[~idx.index.duplicated()]
    print(f"index hours: {len(idx)}  (expect ~8760)")
    for c in ["Tgr", "Tgeo"]:
        print(f"  {c}: valid {idx[c].notna().sum()} h, "
              f"median {idx[c].median():.2f}, range {idx[c].min():.2f}..{idx[c].max():.2f}")

    # --- 2. chain-average hourly Pc5 index from 13 stations ---
    per = {}
    for st in STATIONS:
        p = f"{DATA_DIR}/{st}_2015_1min.csv.gz"
        if not os.path.exists(p):
            print(f"  !! missing {st}")
            continue
        df = pd.read_csv(p, index_col=0, parse_dates=[0])[["Pc5"]]
        per[st] = np.log10(df["Pc5"].clip(lower=1e-6))
    chain_1min = pd.concat(per, axis=1).mean(axis=1)          # mean log-power
    chain_h = chain_1min.resample("1h").mean().dropna()       # hourly mean
    print(f"chain hourly hours: {len(chain_h)}")

    # --- 3. correlations ---
    out = []
    for col in ["Tgr", "Tgeo"]:
        j = pd.concat([chain_h.rename("chain"), idx[col].rename("idx")],
                      axis=1, join="inner").dropna()
        r, pr = pearsonr(j["chain"], j["idx"])
        rho, ps = spearmanr(j["chain"], j["idx"])
        out.append(dict(index=col, n_hours=len(j),
                        pearson_r=r, pearson_p=pr,
                        spearman_rho=rho, spearman_p=ps))
        print(f"\n== chain vs {col} ==  n={len(j)} hours")
        print(f"   Pearson  r = {r:+.3f}  (p={pr:.2e})")
        print(f"   Spearman rho= {rho:+.3f}  (p={ps:.2e})")

    pd.DataFrame(out).to_csv(OUT, index=False)
    print("\nsaved:", OUT)


if __name__ == "__main__":
    main()
