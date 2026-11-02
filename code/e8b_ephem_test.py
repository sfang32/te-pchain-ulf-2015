# -*- coding: utf-8 -*-
"""E8b: RBSP-B 2015 星历 -> 5-min 位置 + 偶极磁纬/MLT/L（测试版：单日抽查）
输出: 控制台抽查 + data/RBSPB_ephem_2015.csv（全年前 3 天示例）
"""
import glob
import numpy as np
import pandas as pd
import spiceypy as sp

EPH = "/data/ssd-stor/OL-CB/DataCollection/AS-01-024/rbsp/rbspb/ephemeris"
POLE_LAT, POLE_LON = 80.0, -72.2
RE = 6371.0

sp.furnsh(f"{EPH}/leap_second_kernel/naif0012.tls")
sp.furnsh(f"{EPH}/planetary_ephemeris/de421.bsp")
for fk in glob.glob(f"{EPH}/frame_kernel/*.tf"):
    sp.furnsh(fk)

spks = []
for f in glob.glob(f"{EPH}/ephemerides/*.bsp"):
    name = f.split("/")[-1]
    try:
        y2 = int(name.replace(".deph.bsp", "").split("_")[3])
    except Exception:
        continue
    if y2 >= 2015:
        spks.append(f); sp.furnsh(f)
print("SPK loaded:", len(spks))
try:
    bid = sp.bodn2c("RBSP_B"); print("RBSP_B id:", bid)
except Exception as e:
    print("bodn2c err:", e); bid = -99


def gmst_rad(et):
    """IAU1982 GMST (radians) from ET (treat UTC~UT1, 误差<<1 deg OK)"""
    jd_ut1 = 2440587.5 + et / 86400.0 + 2415020.0 - 2437684.5  # 不可靠, 改用标准公式
    # 用 spice 自带 et2lst 更稳:
    hh, mm, ss, timstr = sp.et2lst(et, 399, 0.0, "planeto")  # lon=0 -> 本Greenwich 时
    return (hh * 3600 + mm * 60 + ss) / 240.0 * np.pi / 180 * 15 * 0  # placeholder


def gmst2(et):
    # 标准 IAU-1982 GMST（秒->弧度）
    jd = 2451545.0 + et / 86400.0
    T = (jd - 2451545.0) / 36525.0
    g = (67310.54841 + (876600 * 3600 + 8640184.812866) * T + 0.093104 * T**2 - 6.2e-6 * T**3) % 86400
    return np.deg2rad(g / 240.0 * 15 / 15)  # g 为秒 -> deg/240? 修正: g 秒 * (360/86400) deg


def gmst3(et):
    jd = 2451545.0 + et / 86400.0
    T = (jd - 2451545.0) / 36525.0
    gsec = 67310.54841 + (876600 * 3600 + 8640184.812866) * T + 0.093104 * T**2 - 6.2e-6 * T**3
    gdeg = (gsec % 86400) * (360.0 / 86400.0)
    return np.deg2rad(gdeg)


def latlon(pos_j2000, et):
    g = gmst3(et)
    x, y, z = pos_j2000
    xf = x * np.cos(g) + y * np.sin(g)
    yf = -x * np.sin(g) + y * np.cos(g)
    r = np.sqrt(xf**2 + yf**2 + z**2)
    lat = np.degrees(np.arcsin(z / r))
    lon = np.degrees(np.arctan2(yf, xf))
    return lat, ((lon + 180) % 360) - 180, r


def mlat_mlt(lat, lon, r_re, sun_lon):
    """偶极磁纬（对磁北极的大圆角距）与 MLT（卫星磁经线相对太阳磁经线）"""
    la1, lo1 = np.radians(POLE_LAT), np.radians(POLE_LON)
    la2, lo2 = np.radians(lat), np.radians(lon)
    d = np.arccos(np.clip(np.sin(la1) * np.sin(la2) + np.cos(la1) * np.cos(la2) * np.cos(lo2 - lo1), -1, 1))
    mlat = np.degrees(d) - 90.0 if False else 90.0 - np.degrees(d)
    mlat = mlat * (1 if lat >= 0 else -1)
    # MLT: 太阳磁经线为 0；卫星磁经线 = 地理经度相对磁极的方位差（近似：经度差校正有限，直接用 lon-sun_lon 的磁修正略去）
    mlt = ((lon - sun_lon) % 360) / 15.0
    return mlat, mlt


idx = pd.date_range("2015-03-17 00:00", "2015-03-17 23:55", freq="5min")
rows = []
for t in idx:
    et = sp.str2et(t.strftime("%Y-%m-%d %H:%M:%S UTC"))
    try:
        st, _ = sp.spkezr("RBSP_B", et, "J2000", "NONE", "EARTH")
        lat, lon, r = latlon(np.array(st[:3]), et)
        # 太阳地理经度（近似 subsolar lon = 180 - UT*15）
        sun_lon = (180.0 - (t.hour * 60 + t.minute) / 4.0) % 360 - 180
        mla, mlt = mlat_mlt(lat, lon, r / RE, sun_lon)
        L = (r / RE) / np.cos(np.radians(abs(mla)))**2 if abs(mla) < 89 else np.nan
        rows.append((t, lat, lon, r / RE, mla, mlt, L))
    except Exception as e:
        rows.append((t, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
d = pd.DataFrame(rows, columns=["time", "glat", "glon", "r_re", "mlat", "MLT", "L"]).set_index("time")
print(d.head(3).round(2).to_string())
print(d.describe().round(2).loc[["min", "50%", "max"]].to_string())
print("valid:", len(d.dropna()), "/", len(d))
