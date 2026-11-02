# -*- coding: utf-8 -*-
"""Smoke test: 数据管线可行性验证（单台站单日）
1. 拼接某日 10-min 窗口 1Hz 数据 -> 连续 1Hz 序列
2. 滑动 1-min 窗计算 Pc5 (2-7 mHz) 带限功率 -> destination 序列
3. 读取同日 OMNI 1-min -> source 序列，对齐
"""
import glob
import sys
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

STATION = sys.argv[1] if len(sys.argv) > 1 else "MHT"
DATE = sys.argv[2] if len(sys.argv) > 2 else "2015-03-17"  # St. Patrick storm
BASE = f"/data/ssd-stor/fsf/workspaces/research_ulf/datas/dataset_pc/ZW/{STATION}/2015/10min/feature_1Hz"
OMNI_DIR = "/data/ssd-stor/fsf/workspaces/research_ulf/submit_yuanxinxihua/data_omni"

# --- 1. 拼接 1Hz 数据 ---
month = DATE[5:7]
pattern = f"{BASE}/{month}/{DATE}*.csv"
files = sorted(glob.glob(pattern))
print(f"[1] {STATION} {DATE}: {len(files)} 个 10-min 窗口文件")
if not files:
    sys.exit("no files")

dfs = [pd.read_csv(f, index_col=0, parse_dates=[0]) for f in files]
ser = pd.concat(dfs).sort_index()
# 检查连续性
dt = ser.index.to_series().diff().dropna()
gaps = dt[dt != pd.Timedelta("1s")]
print(f"[1] 总点数={len(ser)}, 时间跨度 {ser.index[0]} ~ {ser.index[-1]}, 缺口数={len(gaps)}")
if len(gaps):
    print(gaps.head())

# --- 2. Pc5 带限功率 ---
fs = 1.0
b, a = butter(4, [2/500, 7/500], btype="band")  # 2-7 mHz -> 0.002-0.007 Hz (Nyquist 0.5)
h = ser["H"].interpolate().values
hp = filtfilt(b, a, h - h.mean())
s = pd.Series(hp, index=ser.index)
p1min = (s**2).resample("1min").mean()  # 1-min 窗平均功率
print(f"[2] 1-min Pc5 功率序列: {len(p1min)} 点, median={p1min.median():.4g}")

# --- 3. OMNI 对齐 ---
omni_file = f"{OMNI_DIR}/omni_{DATE.replace('-','')}000000.csv"
omni = pd.read_csv(omni_file, index_col=0, parse_dates=[0])
cols = ["Bz_GSM", "Flow_speed", "Flow_pressure", "Electric_field", "AE-index", "SYM-H"]
print(f"[3] OMNI: {len(omni)} 点, 列齐全: {[c for c in cols if c in omni.columns]}")
df = pd.DataFrame({"Pc5_power": p1min}).join(omni[cols], how="inner")
print(f"[3] 对齐后: {len(df)} 点 (1-min cadence)")
print(df.describe().loc[["mean", "50%"], :].T)
print("SMOKE TEST OK")
