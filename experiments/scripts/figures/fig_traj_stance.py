#!/usr/bin/env python3
"""立場の強弱(5段階)が toorPIA マップ上で『連続グラデーション』として並ぶかを検証する。
（ARI=1.00 の離散分離では出てこない「内部状態が確信度/立場を連続的にエンコードするか」の主張）

Qwen3-4B に原発の是非を、強い反対(0)→反対(1)→中立(2)→賛成(3)→強い賛成(4) のペルソナで各40回。
全トークン(smooth5)を toorPIA に投入。
 左: 全点の密度ヒートマップ（構造の文脈）
 右: 各回答の重心を立場レベルで連続着色（青=反対〜赤=賛成）＋レベル別重心。
連続性 = レベルと「立場軸への射影」の Spearman、およびレベル別平均射影の単調性で評価。

事前(fit時): source ~/work/toorpia/samples/env.sh; export TOORPIA_API_URL=http://localhost:3000
usage: python figures/fig_traj_stance.py [--refit]
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_traj_dna import load_all_variants, make_basemap, HEAT_CMAP  # noqa: E402
from fig_traj import moving_average  # noqa: E402

plt.rcParams["font.family"] = "IPAexGothic"
plt.rcParams["text.parse_math"] = False
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "results", "Qwen3-4B")
LEVEL = {"s0_strong_anti": 0, "s1_anti": 1, "s2_neutral": 2, "s3_pro": 3, "s4_strong_pro": 4}
LNAME = ["強い反対", "反対", "中立", "賛成", "強い賛成"]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--refit", action="store_true")
    args = ap.parse_args()
    meta = json.load(open(os.path.join(RES, "traj_meta_stance.json")))
    cache = os.path.join(RES, "traj_stance_xy.npy")
    # 文書順を固定: (sent_id, variant) を load_all_variants の sorted と合わせるため、
    # ここでは sent_id ごとに variant 昇順で並べ直し、その順で fit する。
    order = sorted(range(len(meta)), key=lambda i: (meta[i]["sent_id"], meta[i]["variant"]))
    meta = [meta[i] for i in order]
    lev = np.array([LEVEL[m["sent_id"]] for m in meta])
    if args.refit or not os.path.exists(cache):
        # load_all_variants は sent_id 単位。全 sent_id を結合（各 sent_id 内 variant 昇順）。
        segs = []
        for sid in sorted(LEVEL):
            allX = load_all_variants(os.path.join(RES, "traj_vectors_stance.csv"), sent_id=sid)
            for v in sorted(allX):
                segs.append(moving_average(allX[v], 5))
        XY = make_basemap(np.vstack(segs), cache, recompute=True)
        np.save(os.path.join(RES, "traj_stance_sizes.npy"), np.array([len(s) for s in segs]))
    else:
        XY = np.load(cache)
    sizes = np.load(os.path.join(RES, "traj_stance_sizes.npy"))
    bounds = np.cumsum([0] + list(sizes))
    assert bounds[-1] == len(XY), (bounds[-1], len(XY))
    curves = [XY[bounds[i]:bounds[i + 1]] for i in range(len(meta))]
    C = np.array([c.mean(0) for c in curves])     # 文書重心

    # 立場軸: level を (x,y) に最小二乗回帰した方向ベクトル
    A = np.c_[C, np.ones(len(C))]
    coef, *_ = np.linalg.lstsq(A, lev, rcond=None)
    w = coef[:2]; w = w / (np.linalg.norm(w) + 1e-9)
    proj = C @ w
    rho = spearman(lev, proj)
    print(f"連続性 Spearman(立場レベル, 立場軸への射影) = {rho:.3f}")
    print("レベル別 平均射影（単調なら連続グラデーション）:")
    means = [proj[lev == k].mean() for k in range(5)]
    for k in range(5):
        print(f"  {k} {LNAME[k]:6s}: {means[k]:+.3f}")
    mono = all(means[k] < means[k + 1] for k in range(4)) or all(means[k] > means[k + 1] for k in range(4))
    print("  → 単調順序:", "はい(連続)" if mono else "いいえ")

    # 図
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 8))
    # 左: 密度ヒートマップ
    a1.set_facecolor("black")
    xmin, ymin = XY.min(0); xmax, ymax = XY.max(0)
    mx, my = (xmax - xmin) * .05, (ymax - ymin) * .05
    ext = [xmin - mx, xmax + mx, ymin - my, ymax + my]
    H, _, _ = np.histogram2d(XY[:, 0], XY[:, 1], bins=440, range=[[ext[0], ext[1]], [ext[2], ext[3]]])
    Zn = gaussian_filter(H.T, sigma=8); Zn /= Zn.max()
    rgba = HEAT_CMAP(np.clip(Zn ** 0.7, 0, 0.82)); rgba[..., 3] = np.clip(Zn ** 0.35, 0, 0.9)
    a1.imshow(rgba, origin="lower", extent=ext, aspect="auto", interpolation="bilinear")
    a1.set_xlim(ext[0], ext[1]); a1.set_ylim(ext[2], ext[3]); a1.set_aspect("equal")
    a1.set_xticks([]); a1.set_yticks([])
    a1.set_title("① 全点の密度ヒートマップ（構造の文脈）", fontsize=12, weight="bold")

    # 右: 文書重心を立場レベルで連続着色
    for i in range(len(meta)):
        c = curves[i]
        a2.plot(c[:, 0], c[:, 1], "-", color=plt.cm.coolwarm(lev[i] / 4), lw=0.4, alpha=0.12, zorder=1)
    sc = a2.scatter(C[:, 0], C[:, 1], c=lev, cmap="coolwarm", s=55, edgecolors="k",
                    linewidths=0.4, zorder=3, vmin=0, vmax=4)
    for k in range(5):
        ck = C[lev == k].mean(0)
        a2.scatter([ck[0]], [ck[1]], marker="*", s=520, color=plt.cm.coolwarm(k / 4),
                   edgecolors="black", linewidths=1.8, zorder=5)
        a2.annotate(str(k), (ck[0], ck[1]), ha="center", va="center", fontsize=11,
                    weight="bold", zorder=6)
    cb = fig.colorbar(sc, ax=a2, fraction=0.046, pad=0.02, ticks=range(5))
    cb.ax.set_yticklabels(LNAME); cb.set_label("立場（ペルソナ）")
    a2.set_aspect("equal"); a2.set_xticks([]); a2.set_yticks([])
    a2.set_title(f"② 各回答の重心を立場で連続着色（★=レベル別重心 0〜4）\n"
                 f"連続性 Spearman={rho:.2f}（{'単調＝連続グラデーション' if mono else '非単調'}）",
                 fontsize=12, weight="bold")
    fig.suptitle("立場の強弱は内部状態に『連続的に』エンコードされるか — 原発是非・5段ペルソナ×各40回（Qwen3-4B・toorPIA）",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(RES, "traj_stance.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
