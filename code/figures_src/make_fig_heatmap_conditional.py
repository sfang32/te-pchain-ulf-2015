# -*- coding: utf-8 -*-
"""Resubmit v1: regenerate fig4_z_heatmap.png (-> Figure 5) and
fig7_conditional.png (-> Figure 8) with the stale embedded "Fig N:" suptitle
prefixes removed (LaTeX assigns figure numbers; Fig1 split shifted all IDs).

Based on v7_revision/make_fig4_fig7.py; data sources unchanged:
  - server_sync/results_e3/sig_Pc5_5min_*.csv   (z heatmap)
  - server_sync/results_e6c/cond_*.csv          (conditional TE)
Output: resubmit_v1/figures/*.png (overwrite in place)
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"E:\project_ulf_omni\ground_pc5"
E3 = os.path.join(ROOT, "server_sync", "results_e3")
E6C = os.path.join(ROOT, "server_sync", "results_e6c")
OUT = os.path.join(ROOT, "resubmit_v1", "figures")
os.makedirs(OUT, exist_ok=True)

order = ["MHT", "MZL", "NAT", "SSL", "MLS", "PXT", "JFT", "HZT", "SYT",
         "ZQT", "LAT", "SYS", "QZT"]
PNAMES = {"Flow_pressure": "Pdyn", "Bz_GSM": "Bz"}

CODE_MAP = {"MHT": "OMOHE", "MZL": "OMZLN", "NAT": "ONOAN", "SSL": "OCSSL",
            "MLS": "OTANC", "PXT": "OPIXI", "JFT": "OYICH", "HZT": "OHZXH",
            "SYT": "OSHYA", "ZQT": "OZHQI", "LAT": "OLSCG", "SYS": "OLEDO",
            "QZT": "OQIZH"}

approx_st = {"SSL", "QZT", "ZQT"}
_suspect = {"HZT"}
def _lbl(s):
    name = CODE_MAP.get(s, s)
    name = name + (" *" if s in approx_st else "")
    name = name + (" (susp.)" if s in _suspect else "")
    return name

# ---------------- z heatmap (Figure 5) ----------------
e3 = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(E3, "sig_Pc5_5min_*.csv"))])
fig, axes = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True)
for ax, p in zip(axes, ["Flow_pressure", "Bz_GSM"]):
    d = e3[e3.param == p].copy()
    d["ord"] = d.station.map({s: i for i, s in enumerate(order)})
    d = d.dropna(subset=["ord"])
    piv = d.pivot_table(index="station", columns="lag_min", values="z").reindex(order)
    vlim = float(np.nanmax(np.abs(piv.values)))
    im = ax.pcolormesh(piv.columns, np.arange(len(piv)), piv.values,
                       cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    ax.set_yticks(np.arange(len(piv)) + 0.5)
    ax.set_yticklabels([_lbl(s) for s in piv.index], fontsize=7)
    ax.set_ylabel("Station")
    ax.set_title(f"{PNAMES[p]} -> Pc5  (symmetric |z| range = ±{vlim:.1f})", fontsize=9)
    cb = plt.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("z (shift-null)")
    ax.text(0.01, 0.96, "* approx. magnetic latitude\nOHZXH: data-quality suspect",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))
axes[-1].set_xlabel("Lag (min)")
plt.suptitle("Significance of information flow, station x lag", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig4_z_heatmap.png"), dpi=150)
plt.close()
print("fig4_z_heatmap regenerated (suptitle without number)")

# ---------------- conditional TE (Figure 8) ----------------
stations = ["MHT", "PXT", "SYS"]

def get_block(src, cond):
    out = []
    for st in stations:
        d = pd.read_csv(os.path.join(E6C, f"cond_{st}.csv"))
        out.append(d[(d.source == src) & (d.conditional == cond)])
    return pd.concat(out, ignore_index=True)

pa = get_block("Flow_pressure", "Bz_GSM")
pb = get_block("Bz_GSM", "Flow_pressure")

def boot_ci(vals, n=10000):
    vals = np.asarray(vals, float)
    if len(vals) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(0)
    bs = np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, data, title in [(axes[0], pa, "Pdyn -> Pc5"),
                        (axes[1], pb, "Bz -> Pc5")]:
    gu = data.groupby("lag_min")["excess_unc"].agg(["mean", "std"])
    gc = data.groupby("lag_min")["excess_cond"].agg(["mean", "std"])
    lag = gu.index.astype(float)
    ax.errorbar(lag, gu["mean"], yerr=gu["std"], fmt="o-", color="#d62728",
                label="unconditional", capsize=2, ms=4)
    ax.errorbar(lag, gc["mean"], yerr=gc["std"], fmt="s--", color="#1f77b4",
                label="conditioned", capsize=2, ms=4)
    ci_lines = []
    for kl in (20, 60, 120):
        if kl in gu.index:
            lo, hi = boot_ci(data[data.lag_min == kl]["excess_unc"].values)
            ci_lines.append(f"lag {kl} min: [{lo:.3f}, {hi:.3f}]")
    if ci_lines:
        ax.text(0.02, 0.02, "95% CI (uncond.):\n" + "\n".join(ci_lines),
                transform=ax.transAxes, fontsize=7, color="#d62728",
                va="bottom", ha="left",
                bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.85))
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Lag (min)")
    ax.set_ylabel("Excess TE (nats)")
    ax.legend(fontsize=8, loc="upper right")
fig.suptitle("Conditional TE channel separation (mean ± 1 std across OMOHE, OPIXI, OLEDO; red box: 95% CI of unconditional excess via station bootstrap)", fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(os.path.join(OUT, "fig7_conditional.png"), dpi=150)
plt.close()
print("fig7_conditional regenerated (suptitle without number)")
