#!/usr/bin/env python3
"""genuine な意味的2パターン（猫推し vs 犬推し）が、同じヒートマップ/軌跡マップ上で
分離するかを検証する。DNA（意味が均一で分離しなかった）との対比。

DeepSeek-R1-Distill-Qwen-1.5B に「猫か犬か、一語で始めて2-3文で論じよ」を100回。
全トークン(smooth5)を toorPIA に投入 → スタンス(Cat/Dog/Other)で色分け描画 ＋
教師なしk=2クラスタリングが Cat/Dog を復元するか(ARI)を測る。

事前(fitする時のみ): source ~/work/toorpia/samples/env.sh; export TOORPIA_API_URL=http://localhost:3000
usage: python figures/fig_traj_catdog.py [--refit]
"""
import argparse
import json
import os
import re
import sys

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
STC = {"Cat": "#2fe0e0", "Dog": "#ff5fd0", "Other": "#9aa0a6"}   # 猫=シアン/犬=マゼンタ/他=灰


def stance(t):
    w = re.sub(r"[*_#`'\"]", "", t).strip().lower().split()
    h = w[0].rstrip(",.:") if w else ""
    return "Cat" if h in ("cat", "cats") else "Dog" if h in ("dog", "dogs") else "Other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(RES, "traj_meta_catdog.json")))
    cache = os.path.join(RES, "traj_catdog_xy.npy")
    if args.refit or not os.path.exists(cache):
        allX = load_all_variants(os.path.join(RES, "traj_vectors_catdog.csv"), sent_id="catdog")
        X = np.vstack([moving_average(allX[v], 5) for v in sorted(allX)])
        XY = make_basemap(X, cache, recompute=True)
    else:
        XY = np.load(cache)
    bounds = np.cumsum([0] + [m["n_tokens"] for m in meta])
    assert bounds[-1] == len(XY), (bounds[-1], len(XY))
    curves = [XY[bounds[i]:bounds[i + 1]] for i in range(len(meta))]
    st = [stance(m["text"]) for m in meta]
    from collections import Counter
    cnt = Counter(st)
    print("stance:", dict(cnt))

    # --- 分離度: 文書重心の教師なしk=2 が Cat/Dog を復元するか ---
    C = np.array([curves[i].mean(0) for i in range(len(meta))])
    cd = [i for i in range(len(meta)) if st[i] in ("Cat", "Dog")]
    Ccd = C[cd]
    true = np.array([0 if st[i] == "Cat" else 1 for i in cd])
    km = KMeans(2, n_init=20, random_state=0).fit_predict(Ccd)
    ari = adjusted_rand_score(true, km)
    cm = confusion_matrix(true, km)
    # 教師ありの分離（重心距離）も
    cenC = C[[i for i in range(len(meta)) if st[i] == "Cat"]].mean(0)
    cenD = C[[i for i in range(len(meta)) if st[i] == "Dog"]].mean(0)
    within = np.mean([np.linalg.norm(C[i] - (cenC if st[i] == "Cat" else cenD))
                      for i in cd])
    sep = float(np.linalg.norm(cenC - cenD) / (within + 1e-9))
    print(f"教師なしk=2 ARI(Cat vs Dog) = {ari:.3f}  混同={cm.tolist()}")
    print(f"Cat/Dog 重心間距離 / 群内広がり = {sep:.2f}")

    # --- 図: ヒートマップ(全点) + スタンス色分け軌跡 ---
    fig, ax = plt.subplots(figsize=(11.5, 10))
    fig.patch.set_facecolor("black"); ax.set_facecolor("black")
    xmin, ymin = XY.min(0); xmax, ymax = XY.max(0)
    mx, my = (xmax - xmin) * .05, (ymax - ymin) * .05
    ext = [xmin - mx, xmax + mx, ymin - my, ymax + my]
    H, _, _ = np.histogram2d(XY[:, 0], XY[:, 1], bins=440,
                             range=[[ext[0], ext[1]], [ext[2], ext[3]]])
    Zn = gaussian_filter(H.T, sigma=8); Zn /= Zn.max()
    rgba = HEAT_CMAP(np.clip(Zn ** 0.7, 0, 0.82))
    rgba[..., 3] = np.clip(Zn ** 0.35, 0, 0.9)
    ax.imshow(rgba, origin="lower", extent=ext, aspect="auto", interpolation="bilinear", zorder=0)

    for i in range(len(meta)):
        c = curves[i]
        ax.plot(c[:, 0], c[:, 1], "-", color=STC[st[i]], lw=0.7, alpha=0.32, zorder=2)
    for i in range(len(meta)):
        ax.scatter([curves[i][0, 0]], [curves[i][0, 1]], marker="o", s=22,
                   color=STC[st[i]], edgecolors="white", linewidths=0.3, alpha=0.9, zorder=4)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    handles = [Line2D([0], [0], color=STC[k], lw=3, label=f"{k} n={cnt.get(k,0)}")
               for k in ["Cat", "Dog", "Other"] if cnt.get(k, 0)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=3, fontsize=11, frameon=True, facecolor="black",
              edgecolor="white", labelcolor="white")
    ax.set_title("『猫 vs 犬どちらが良いペット?』を100回（DeepSeek-R1-1.5B）— 意味的2パターン\n"
                 f"全{len(XY)}点の密度ヒートマップ + スタンス別軌跡（toorPIA）  "
                 f"教師なしARI(Cat/Dog)={ari:.2f}・分離比={sep:.1f}",
                 fontsize=12, weight="bold", color="white")
    fig.tight_layout()
    out = os.path.join(RES, "traj_catdog.png")
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="black")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
