#!/usr/bin/env python3
"""DeepSeek-R1-1.5B「水は100℃で沸騰?」100回。回答が『気圧に言及する(条件付き)』と
『単純Yes(気圧なし)』に割れる（89/11）。この内容差がマップ上で分離するかを見る。

cat/dog（概念が近く差が1語）と違い、ここでは差分が "standard atmospheric pressure…" という
実質的なトークン列として存在する → 分離しやすいはず、という仮説の検証。

事前(fit時): source ~/work/toorpia/samples/env.sh; export TOORPIA_API_URL=http://localhost:3000
usage: python figures/fig_traj_water.py [--refit]
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_traj_dna import load_all_variants, make_basemap, HEAT_CMAP  # noqa: E402
from fig_traj import moving_average  # noqa: E402

plt.rcParams["font.family"] = "IPAexGothic"
plt.rcParams["text.parse_math"] = False
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "results", "DeepSeek-R1-Distill-Qwen-1.5B")
COL = {"気圧あり": "#2fe0e0", "気圧なし": "#ff5fd0"}


def is_caveat(t):
    low = t.lower()
    return any(k in low for k in ["pressure", "atm", "atmospher", "sea level",
                                  "altitude", "101.3", "100 kpa", "760", "standard cond"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true")
    args = ap.parse_args()
    meta = json.load(open(os.path.join(RES, "traj_meta_water.json")))
    cache = os.path.join(RES, "traj_water_xy.npy")
    if args.refit or not os.path.exists(cache):
        allX = load_all_variants(os.path.join(RES, "traj_vectors_water.csv"), sent_id="water")
        X = np.vstack([moving_average(allX[v], 5) for v in sorted(allX)])
        XY = make_basemap(X, cache, recompute=True)
    else:
        XY = np.load(cache)
    bounds = np.cumsum([0] + [m["n_tokens"] for m in meta])
    assert bounds[-1] == len(XY), (bounds[-1], len(XY))
    curves = [XY[bounds[i]:bounds[i + 1]] for i in range(len(meta))]
    lab = ["気圧あり" if is_caveat(m["text"]) else "気圧なし" for m in meta]
    cnt = Counter(lab)
    print("split:", dict(cnt))

    # 分離度: 文書重心 と 終端点（差は末尾に出るので末尾も）で教師なしk=2 vs caveat/bare
    Cc = np.array([curves[i].mean(0) for i in range(len(meta))])
    Ce = np.array([curves[i][-1] for i in range(len(meta))])
    true = np.array([0 if lab[i] == "気圧あり" else 1 for i in range(len(meta))])
    for name, P in [("文書重心", Cc), ("終端点", Ce)]:
        km = KMeans(2, n_init=20, random_state=0).fit_predict(P)
        ari = adjusted_rand_score(true, km)
        # bare(少数)が固まっているか: bare重心への距離で caveat と分離するか
        cB = P[true == 1].mean(0); cA = P[true == 0].mean(0)
        win = np.mean([np.linalg.norm(P[i] - (cA if true[i] == 0 else cB)) for i in range(len(P))])
        sep = float(np.linalg.norm(cA - cB) / (win + 1e-9))
        print(f"  [{name}] 教師なしARI={ari:+.3f}  気圧あり/なし重心分離比={sep:.2f}  混同={confusion_matrix(true,km).tolist()}")

    # 図
    fig, ax = plt.subplots(figsize=(11.5, 10))
    fig.patch.set_facecolor("black"); ax.set_facecolor("black")
    xmin, ymin = XY.min(0); xmax, ymax = XY.max(0)
    mx, my = (xmax - xmin) * .05, (ymax - ymin) * .05
    ext = [xmin - mx, xmax + mx, ymin - my, ymax + my]
    H, _, _ = np.histogram2d(XY[:, 0], XY[:, 1], bins=440, range=[[ext[0], ext[1]], [ext[2], ext[3]]])
    Zn = gaussian_filter(H.T, sigma=8); Zn /= Zn.max()
    rgba = HEAT_CMAP(np.clip(Zn ** 0.7, 0, 0.82)); rgba[..., 3] = np.clip(Zn ** 0.35, 0, 0.9)
    ax.imshow(rgba, origin="lower", extent=ext, aspect="auto", interpolation="bilinear", zorder=0)
    # 多数派(気圧あり)→少数派(気圧なし)の順、少数派を上に
    order = sorted(range(len(meta)), key=lambda i: 0 if lab[i] == "気圧あり" else 1)
    for i in order:
        c = curves[i]
        ax.plot(c[:, 0], c[:, 1], "-", color=COL[lab[i]], lw=0.8,
                alpha=0.30 if lab[i] == "気圧あり" else 0.85, zorder=2 + (lab[i] == "気圧なし") * 3)
    for i in order:
        c = curves[i]
        ax.scatter([c[0, 0]], [c[0, 1]], marker="o", s=20 if lab[i] == "気圧あり" else 55,
                   color=COL[lab[i]], edgecolors="white", linewidths=0.3, alpha=0.9,
                   zorder=4 + (lab[i] == "気圧なし") * 3)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    handles = [Line2D([0], [0], color=COL[k], lw=3, label=f"{k} n={cnt.get(k,0)}") for k in COL if cnt.get(k, 0)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
              fontsize=11, frameon=True, facecolor="black", edgecolor="white", labelcolor="white")
    ax.set_title("『水は100℃で沸騰?』DeepSeek-R1-1.5B 100回 — 気圧条件あり vs 単純Yes\n"
                 f"全{len(XY)}点の密度ヒートマップ + パターン別軌跡（toorPIA）",
                 fontsize=12, weight="bold", color="white")
    fig.tight_layout()
    out = os.path.join(RES, "traj_water.png")
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="black")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
