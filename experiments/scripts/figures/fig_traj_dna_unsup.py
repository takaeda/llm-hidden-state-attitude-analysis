#!/usr/bin/env python3
"""ラベル無し（教師なし）で、このマップから意味の2種類を識別できるか？の検証図。

左: 教師なし k=2（BICもk=2を選ぶ）が見つける2群。
右: 本当の意味ラベル（二重/三重）。
両者は一致しない（ARI≈0）＝マップの自然な2群は『言い回し/領域』の差で、意味の差ではない。

入力: results/Qwen3-4B/traj_dna_xy.npy, traj_meta_dna.json
usage: python figures/fig_traj_dna_unsup.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_traj_dna import pattern, PAT  # noqa: E402

plt.rcParams["font.family"] = "IPAexGothic"
plt.rcParams["text.parse_math"] = False
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "results", "Qwen3-4B")


def main():
    XY = np.load(os.path.join(RES, "traj_dna_xy.npy"))
    meta = json.load(open(os.path.join(RES, "traj_meta_dna.json")))
    bounds = np.cumsum([0] + [m["n_tokens"] for m in meta])
    true = np.array([1 if pattern(m["text"]) == "yes_triple" else 0 for m in meta])
    C = np.array([XY[bounds[i]:bounds[i + 1]].mean(0) for i in range(len(meta))])

    km = KMeans(2, n_init=20, random_state=0).fit_predict(C)
    ari = adjusted_rand_score(true, km)
    cm = confusion_matrix(true, km)
    tri_clu = 0 if cm[1, 0] >= cm[1, 1] else 1
    purity = cm[1, tri_clu] / cm[:, tri_clu].sum()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 7.4))
    # 左: 教師なしクラスタ
    for k, col in [(0, "#8c564b"), (1, "#17becf")]:
        m = km == k
        a1.scatter(C[m, 0], C[m, 1], s=70, color=col, edgecolors="white",
                   label=f"教師なしクラスタ{k} (n={m.sum()})")
    a1.set_title("① ラベル無しで見つかる2群（GMM-BICもk=2を選ぶ）\n"
                 "＝言い回し/領域による多数派の分割", fontsize=12, weight="bold")
    a1.legend(loc="best", fontsize=10); a1.set_aspect("equal")
    a1.set_xticks([]); a1.set_yticks([])

    # 右: 真の意味ラベル
    a2.scatter(C[true == 0, 0], C[true == 0, 1], s=70, color=PAT["double"][1],
               edgecolors="white", label=f"二重らせん(正) {(true==0).sum()}本")
    a2.scatter(C[true == 1, 0], C[true == 1, 1], s=120, color=PAT["yes_triple"][1],
               edgecolors="black", linewidths=1.1, zorder=5,
               label=f"Yes:三重(誤) {(true==1).sum()}本")
    a2.set_title("② 本当の意味ラベル（二重 / 三重）\n"
                 "三重は上に偏るが純然たる『塊』ではない", fontsize=12, weight="bold")
    a2.legend(loc="best", fontsize=10); a2.set_aspect("equal")
    a2.set_xticks([]); a2.set_yticks([])

    fig.suptitle(f"ラベル無しでは意味の2種類を識別できない： 教師なしk=2 vs 意味ラベルの一致 ARI={ari:.2f}（≈0）"
                 f"／三重寄りクラスタの純度{purity:.0%}　混同{cm.tolist()}",
                 fontsize=12.5, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(RES, "traj_dna_unsup.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"ARI={ari:.3f}  三重寄りクラスタ純度={purity:.0%}  混同={cm.tolist()}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
