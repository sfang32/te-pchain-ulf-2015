# -*- coding: utf-8 -*-
"""E1: TE 管线验证
(a) 耦合 AR 模型 ground truth：TE(x->y) 应显著 > TE(y->x)
(b) JIDT vs infomeasure 交叉验证
(c) 零假设标定：无耦合对 + 时间打乱 surrogate 的 95%/99% 阈值（n=2000, 100 次）
输出: results/e1_validation.csv + 控制台报告
"""
import jpype
import numpy as np
import pandas as pd

JAR = "/data/ssd-stor/fsf/softwares/infodynamics-dist-1.5/infodynamics.jar"
OUT = "/data/ssd-stor/fsf/workspaces/research_ulf/jb_te_jastp/results/e1_validation.csv"


def init_jvm():
    if not jpype.isJVMStarted():
        jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + JAR)


def te_jidt(source, dest, tau=0, k=1, l=1, knn=4):
    if tau > 0:
        source, dest = source[:-tau], dest[tau:]
    calc = jpype.JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov()
    calc.setProperty("k_HISTORY", str(k))
    calc.setProperty("l_HISTORY", str(l))
    calc.setProperty("k", str(knn))
    calc.setProperty("NOISE_LEVEL_TO_ADD", "1e-8")
    calc.initialise()
    calc.setObservations(np.asarray(source, float), np.asarray(dest, float))
    return float(calc.computeAverageLocalOfObservations())


def te_infomeasure(source, dest, tau=0, k=4):
    try:
        import infomeasure as im
    except ImportError:
        return float("nan")
    if tau > 0:
        source, dest = source[:-tau], dest[tau:]
    return float(im.transfer_entropy(np.asarray(source, float), np.asarray(dest, float),
                                     approach="kraskov", k=k, lag=0))


def z(x):
    return (x - np.nanmean(x)) / np.nanstd(x)


def coupled_ar(n=4000, a=0.6, delay=5, seed=42):
    rng = np.random.default_rng(seed)
    x = np.zeros(n); y = np.zeros(n)
    e = rng.normal(size=n); u = rng.normal(size=n)
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + e[t]
        y[t] = 0.8 * y[t - 1] + (a * x[t - delay] if t >= delay else 0.0) + u[t]
    return z(x), z(y)


def main():
    init_jvm()
    rng = np.random.default_rng(0)
    rows = []

    # (a) ground truth
    x, y = coupled_ar()
    te_xy = te_jidt(x, y, tau=0)   # delay 已内嵌在 y 构造中
    te_yx = te_jidt(y, x, tau=0)
    rows.append(dict(test="a_ground_truth", item="TE(x->y)_jidt", value=te_xy))
    rows.append(dict(test="a_ground_truth", item="TE(y->x)_jidt", value=te_yx))
    print(f"(a) TE(x->y)={te_xy:.4f} nats  vs  TE(y->x)={te_yx:.4f}  -> {te_xy > te_yx}")

    # (b) cross-library
    te_xy_im = te_infomeasure(x, y)
    rows.append(dict(test="b_cross_library", item="TE(x->y)_infomeasure", value=te_xy_im))
    rows.append(dict(test="b_cross_library", item="abs_diff", value=abs(te_xy - te_xy_im)))
    print(f"(b) infomeasure TE(x->y)={te_xy_im:.4f}  |diff|={abs(te_xy-te_xy_im):.4f}")

    # (c) null calibration at n=2000
    n, n_sur = 2000, 100
    null_vals, shuf_vals = [], []
    for i in range(n_sur):
        a = rng.normal(size=n)
        b = rng.normal(size=n)  # 独立对
        null_vals.append(te_jidt(a, b))
        s = z(np.concatenate([np.sin(2 * np.pi * np.arange(n) / 50) + 0.1 * rng.normal(size=n)]))
        t = 0.7 * np.roll(s, -7) + 0.3 * rng.normal(size=n)
        perm = rng.permutation(n)
        shuf_vals.append(te_jidt(s, z(t)[perm]))  # 打乱 dest 破坏耦合
    p95, p99 = np.percentile(null_vals, [95, 99])
    s95, s99 = np.percentile(shuf_vals, [95, 99])
    rows.append(dict(test="c_null_iid_n2000", item="p95", value=p95))
    rows.append(dict(test="c_null_iid_n2000", item="p99", value=p99))
    rows.append(dict(test="c_shuffle_dest", item="p95", value=s95))
    rows.append(dict(test="c_shuffle_dest", item="p99", value=s99))
    print(f"(c) iid null: p95={p95:.4f} p99={p99:.4f};  shuffle-dest: p95={s95:.4f} p99={s99:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print("E1 DONE ->", OUT)


if __name__ == "__main__":
    main()
