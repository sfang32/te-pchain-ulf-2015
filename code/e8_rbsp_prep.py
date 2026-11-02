# -*- coding: utf-8 -*-
"""E8-prep: RBSP-B Pc5 功率序列（L1 标注 IP -> 5-min）+ OMNI 合并
输出: jb_te_jastp/data/RBSPB_2015_1min.csv.gz（5-min cadence, 列 Pc5/coverage + OMNI, 格式与台站文件一致,
      可直接被 e2/e3/e4 管线以 station=RBSPB 调用）
注意: 卫星为移动平台（L/MLT 随轨道变化），TE 结果解释需注明此混杂；后续可按 MLT 分层。
"""
import glob
import numpy as np
import pandas as pd

ROOT = "/data/ssd-stor/fsf/workspaces/research_ulf"
L1_DIR = f"{ROOT}/jb_detection_rbsp_pc/dataset_b/L1/pc5/2015"
OMNI_DIR = f"{ROOT}/submit_yuanxinxihua/data_omni"
OUT = f"{ROOT}/jb_te_jastp/data/RBSPB_2015_1min.csv.gz"
OMNI_COLS = ["Bz_GSM", "Flow_speed", "Flow_pressure", "Electric_field",
             "AE-index", "AL-index", "AU-index", "SYM-H", "ASY-H"]

files = sorted(glob.glob(f"{L1_DIR}/*/*.csv"))
print(f"L1 files: {len(files)}")
dfs = []
for f in files:
    d = pd.read_csv(f, index_col=0, parse_dates=[0], usecols=[0, 3])  # index + Y/IP
    dfs.append(d)
ser = pd.concat(dfs).sort_index()
if isinstance(ser, pd.DataFrame):
    ser = ser.iloc[:, 0]
ser = ser[~ser.index.duplicated()]
p5 = ser.resample("5min").mean()
cov = ser.notna().resample("5min").mean()
df = pd.DataFrame({"Pc5": p5, "coverage": cov})
print(f"5-min bins: {len(df)}, valid: {df.Pc5.notna().sum()} ({df.Pc5.notna().mean():.1%})")

omni_files = sorted(glob.glob(f"{OMNI_DIR}/omni_2015*.csv"))
omni = pd.concat([pd.read_csv(f, index_col=0, parse_dates=[0]) for f in omni_files]).sort_index()
omni = omni[~omni.index.duplicated()]
for c in OMNI_COLS:
    if c not in omni.columns:
        omni[c] = np.nan
df = df.join(omni[OMNI_COLS], how="inner")
df["station"] = "RBSPB"
df.to_csv(OUT, compression="gzip")
print(f"[OK] -> {OUT}, rows={len(df)}")
