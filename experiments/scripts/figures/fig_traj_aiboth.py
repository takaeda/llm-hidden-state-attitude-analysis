#!/usr/bin/env python3
"""Qwen3-4B「AIは仕事を奪うか? 両論併記で簡潔に」100回。両論併記でも、冒頭で
『奪う側を先(likely to automate…)』か『安全側を先(unlikely to take away…)』かで割れる(83/17)。
差は冒頭に出る → 先頭点で分離するか、全体では収束するかを見る。

事前(fit時): source ~/work/toorpia/samples/env.sh; export TOORPIA_API_URL=http://localhost:3000
usage: python figures/fig_traj_aiboth.py [--refit]
"""
import argparse
import json
import os
import re
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
    os.path.abspath(__file__)))), "results", "Qwen3-4B")
COL = {"奪う側先": "#ff3cd0", "安全側先": "#2fe0e0"}


def lead(t):
    h = re.sub(r"[*_#`]", " ", t).strip().lower()[:70]
    if "unlikely" in h or "will not take" in h or "won't take" in h or ("not " in h and "replace" in h) or "augment human" in h:
        return "安全側先"
    if "likely to" in h or "will automate" in h or "will likely" in h:
        return "奪う側先"
    return "奪う側先"   # 残りは likely 系が大半


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--refit", action="store_true")
    args = ap.parse_args()
    meta = json.load(open(os.path.join(RES, "traj_meta_aiboth.json")))
    cache = os.path.join(RES, "traj_aiboth_xy.npy")
    if args.refit or not os.path.exists(cache):
        allX = load_all_variants(os.path.join(RES, "traj_vectors_aiboth.csv"), sent_id="ai_both")
        X = np.vstack([moving_average(allX[v], 5) for v in sorted(allX)])
        XY = make_basemap(X, cache, recompute=True)
    else:
        XY = np.load(cache)
    bounds = np.cumsum([0] + [m["n_tokens"] for m in meta])
    assert bounds[-1] == len(XY), (bounds[-1], len(XY))
    curves = [XY[bounds[i]:bounds[i + 1]] for i in range(len(meta))]
    lab = [lead(m["text"]) for m in meta]
    cnt = Counter(lab); print("split:", dict(cnt))

    true = np.array([0 if lab[i] == "奪う側先" else 1 for i in range(len(meta))])
    def report(P, name):
        km = KMeans(2, n_init=20, random_state=0).fit_predict(P)
        ari = adjusted_rand_score(true, km)
        cA = P[true == 0].mean(0); cB = P[true == 1].mean(0)
        win = np.mean([np.linalg.norm(P[i] - (cA if true[i] == 0 else cB)) for i in range(len(P))])
        sep = float(np.linalg.norm(cA - cB) / (win + 1e-9))
        print(f"  [{name}] 教師なしARI={ari:+.3f}  奪う/安全 重心分離比={sep:.2f}  混同={confusion_matrix(true,km).tolist()}")
    report(np.array([c[0] for c in curves]), "先頭1トークン")
    report(np.array([c[:3].mean(0) for c in curves]), "先頭3トークン")
    report(np.array([c.mean(0) for c in curves]), "文書重心")

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
    order = sorted(range(len(meta)), key=lambda i: 0 if lab[i] == "奪う側先" else 1)
    for i in order:
        c = curves[i]
        ax.plot(c[:, 0], c[:, 1], "-", color=COL[lab[i]], lw=0.7,
                alpha=0.28 if lab[i] == "奪う側先" else 0.8, zorder=2 + (lab[i] == "安全側先") * 3)
    for i in order:
        c = curves[i]
        ax.scatter([c[0, 0]], [c[0, 1]], marker="o", s=20 if lab[i] == "奪う側先" else 55,
                   color=COL[lab[i]], edgecolors="white", linewidths=0.3, alpha=0.9,
                   zorder=4 + (lab[i] == "安全側先") * 3)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    handles = [Line2D([0], [0], color=COL[k], lw=3, label=f"{k} n={cnt.get(k,0)}") for k in COL if cnt.get(k, 0)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
              fontsize=11, frameon=True, facecolor="black", edgecolor="white", labelcolor="white")
    ax.set_title("『AIは仕事を奪うか? 両論併記』Qwen3-4B 100回 — 冒頭で奪う側先/安全側先に割れる\n"
                 f"全{len(XY)}点の密度ヒートマップ + lead別軌跡（toorPIA）",
                 fontsize=12, weight="bold", color="white")
    fig.tight_layout()
    out = os.path.join(RES, "traj_aiboth.png")
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="black")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
