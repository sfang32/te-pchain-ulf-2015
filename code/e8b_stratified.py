# -*- coding: utf-8 -*-
"""E8b: MLT/L 分层的 TE — 太阳风 -> RBSP-B Pc5 功率（解决移动平台混杂）
方法: 全年星历（5-min）+ L1 Pc5 功率; 对每个 lag, 只取 dest(t+lag) 位于指定 MLT 扇区且 L<6 的样本对,
     TE 一次批量计算（随机子采样上限 8000 点）, null 用同 mask 下循环移位源（12 次）。
扇区: dawn 03-09 / day 09-15 / dusk 15-21 / night 21-03 (MLT, 近似偶极几何)
输出: results/e8/e8b_stratified.csv
"""
import glob
import os
import numpy as np
import pandas as pd
import spiceypy as sp
import jpype

ROOT = "/data/ssd-stor/fsf/workspaces/research_ulf"
DATA_DIR = f"{ROOT}/jb_te_jastp/data"
OUT_DIR = f"{ROOT}/jb_te_jastp/results/e8"
os.makedirs(OUT_DIR, exist_ok=True)
JAR = "/data/ssd-stor/fsf/softwares/infodynamics-dist-1.5/infodynamics.jar"
EPH = "/data/ssd-stor/OL-CB/DataCollection/AS-01-024/rbsp/rbspb/ephemeris"
POLE_LAT, POLE_LON = 80.0, -72.2
RE = 6371.0
PARAMS = ["Flow_pressure", "Bz_GSM"]
LAGS_MIN = [20, 60, 120, 240]
SECTORS = {"dawn": (3, 9), "day": (9, 15), "dusk": (15, 21), "night": (21, 27)}
CAP = 8000
N_SHIFT = 12
CAD = 5


def init_jvm():
    if not jpype.isJVMStarted():
        jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + JAR)


def te_jidt(a, b):
    calc = jpype.JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov()
    calc.setProperty("k_HISTORY", "1"); calc.setProperty("l_HISTORY", "6")
    calc.setProperty("k", "4"); calc.setProperty("NOISE_LEVEL_TO_ADD", "1e-8")
    calc.initialise()
    calc.setObservations(np.asarray(a, float), np.asarray(b, float))
    return float(calc.computeAverageLocalOfObservations())


def zscore(v):
    mu, sd = np.nanmean(v), np.nanstd(v)
    return (v - mu) / (sd if sd > 0 else 1.0)


def mag_lon(lat, lon, pole_lat, pole_lon):
    """点相对偶极磁北极的磁经线（弧度, 相对过北极的地理参考线）"""
    la1, lo1 = np.radians(pole_lat), np.radians(pole_lon)
    la2, lo2 = np.radians(lat), np.radians(lon)
    return np.arctan2(np.sin(lo2 - lo1) * np.cos(la2),
                      np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(lo2 - lo1))


def main():
    # --- 星历 ---
    sp.furnsh(f"{EPH}/leap_second_kernel/naif0012.tls")
    sp.furnsh(f"{EPH}/planetary_ephemeris/de421.bsp")
    for fk in glob.glob(f"{EPH}/frame_kernel/*.tf"):
        sp.furnsh(fk)
    n_spk = 0
    for f in glob.glob(f"{EPH}/ephemerides/*.bsp"):
        name = f.split("/")[-1]
        try:
            if int(name.replace(".deph.bsp", "").split("_")[3]) >= 2015:
                sp.furnsh(f); n_spk += 1
        except Exception:
            pass
    print(f"SPK loaded: {n_spk}", flush=True)

    idx = pd.date_range("2015-01-01 00:00", "2015-12-31 23:55", freq="5min")
    n = len(idx)
    lat = np.full(n, np.nan); lon = np.full(n, np.nan); rre = np.full(n, np.nan)
    ets = [sp.str2et(t.strftime("%Y-%m-%d %H:%M:%S UTC")) for t in idx]
    for i, et in enumerate(ets):
        try:
            st, _ = sp.spkezr("RBSP_B", et, "J2000", "NONE", "EARTH")
            p = np.array(st[:3])
            r = np.linalg.norm(p)
            jd = 2451545.0 + et / 86400.0
            T = (jd - 2451545.0) / 36525.0
            gsec = 67310.54841 + (876600 * 3600 + 8640184.812866) * T + 0.093104 * T**2 - 6.2e-6 * T**3
            g = np.deg2rad((gsec % 86400) * (360.0 / 86400.0))
            x, y, z = p
            xf = x * np.cos(g) + y * np.sin(g)
            yf = -x * np.sin(g) + y * np.cos(g)
            lat[i] = np.degrees(np.arcsin(z / r))
            lon[i] = np.degrees(np.arctan2(yf, xf))
            rre[i] = r / RE
        except Exception:
            continue
    print(f"ephem valid: {np.isfinite(lat).sum()}/{n}", flush=True)

    # 偶极磁纬/MLT
    la1, lo1 = np.radians(POLE_LAT), np.radians(POLE_LON)
    la2 = np.radians(lat); dlon = np.radians(lon - POLE_LON)
    ang = np.arccos(np.clip(np.sin(la1) * np.sin(la2) + np.cos(la1) * np.cos(la2) * np.cos(dlon), -1, 1))
    mlat = np.where(lat >= 0, 90 - np.degrees(ang), -(90 - np.degrees(ang)))
    lam = mag_lon(lat, lon, POLE_LAT, POLE_LON)
    ut_h = idx.hour + idx.minute / 60.0
    sun_lon = np.radians((180.0 - ut_h * 15.0))
    lam_sun = mag_lon(0.0, np.degrees(sun_lon), POLE_LAT, POLE_LON)
    mlt = ((lam - lam_sun) % (2 * np.pi)) * 12 / np.pi
    L = rre / np.cos(np.radians(np.clip(np.abs(mlat), 0, 88)))**2
    emph = pd.DataFrame({"mlat": mlat, "MLT": mlt, "L": L}, index=idx)
    emph.to_csv(f"{DATA_DIR}/RBSPB_ephem_2015.csv")
    print("ephem saved", flush=True)

    # --- Pc5 功率 + OMNI ---
    df = pd.read_csv(f"{DATA_DIR}/RBSPB_2015_1min.csv.gz", index_col=0, parse_dates=[0])
    df = df.drop(columns=["station"], errors="ignore").select_dtypes(include=[np.number])
    dest_full = pd.Series(zscore(np.log10(df["Pc5"].clip(lower=1e-6) + 1e-3).values), index=df.index)
    dest_valid = df["Pc5"].notna().values

    omni = df
    init_jvm()
    rng = np.random.default_rng(2026)
    rows = []
    for param in PARAMS:
        src_all = pd.Series(zscore(omni[param].values), index=omni.index)
        shifts = rng.integers(7 * 24 * 12, 41 * 24 * 12, size=N_SHIFT * len(SECTORS) * len(LAGS_MIN) + 10)
        k = 0
        for sname, (h0, h1) in SECTORS.items():
            # 扇区掩码（基于测量时刻 t+lag 的卫星 MLT/L）
            mlt = emph["MLT"].values
            mlatv = emph["mlat"].values
            Lv = emph["L"].values
            if h0 < h1:
                insec = (mlt >= h0) & (mlt < h1)
            else:
                insec = (mlt >= h0) | (mlt < h1 - 24)
            base = insec & dest_valid & (Lv < 6) & (np.abs(mlatv) < 25)
            for lag_min in LAGS_MIN:
                lag = lag_min // CAD
                sel = np.zeros(n, bool)
                sel[lag:] = base[lag:]  # dest(t+lag) 在扇区
                # 源在该样本时刻的有效性
                for tag, s_arr in [("true", src_all.values)]:
                    pass
                def compute(src_shifted):
                    i = np.where(sel)[0]
                    # dest 下标 = i (t+lag 已含在 sel 定义: sel[i]=base[i] 意味 dest index=i, src index=i-lag)
                    ii = i[(i - lag >= 0)]
                    a = src_shifted[ii - lag]
                    b = dest_full.values[ii]
                    m = ~(np.isnan(a) | np.isnan(b))
                    a, b = a[m], b[m]
                    if len(a) < 400:
                        return np.nan, 0
                    if len(a) > CAP:
                        sub = rng.choice(len(a), CAP, replace=False)
                        a, b = a[sub], b[sub]
                    return te_jidt(a, b), len(a)
                te_true, npts = compute(src_all.values)
                nulls = []
                for _ in range(N_SHIFT):
                    s_r = np.roll(src_all.values, int(shifts[k])); k += 1
                    nv, _ = compute(s_r)
                    nulls.append(nv)
                nulls = np.array(nulls, float); nulls = nulls[~np.isnan(nulls)]
                mu, sd = float(np.mean(nulls)), float(np.std(nulls))
                rows.append(dict(sector=sname, param=param, lag_min=lag_min,
                                 TE_true=te_true, n_points=npts,
                                 null_mean=mu, null_std=sd,
                                 excess=te_true - mu,
                                 z=(te_true - mu) / sd if sd > 0 else np.nan))
                print(f"  [{param} {sname} lag{lag_min}] n={npts} TE={te_true:.4f} null={mu:.4f} z={rows[-1]['z']:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/e8b_stratified.csv", index=False)
    print("E8b DONE", flush=True)


if __name__ == "__main__":
    main()
