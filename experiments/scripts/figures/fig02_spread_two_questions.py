#!/usr/bin/env python3
"""教材図 images/02_spread_two_questions.png を生成する（スライド7）。

「地球は太陽を回る？」(st01) と「DNAは三重らせん？」(sf07) の各12試行の
出だし3トークンhidden state（平均プール）を、toorPIAで2次元化して点群表示。
一貫な質問は一点に重なり、割れる質問は立場ごとの塊に分かれることを示す。

入力: results/Qwen3-4B/full_first3.npz（09_gen_full.py の出力）
      results/Qwen3-4B/full_texts.json（回答文。立場の色分けに使用）
      results/Qwen3-4B/toorpia_pair_first3_xy.npy（toorPIA座標キャッシュ）

usage: python fig02_spread_two_questions.py
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "IPAexGothic"
EXP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(EXP)
RES = os.path.join(EXP, "results", "Qwen3-4B")
OUT = os.path.join(ROOT, "images", "02_spread_two_questions.png")

z = np.load(os.path.join(RES, "full_first3.npz"))
texts = json.load(open(os.path.join(RES, "full_texts.json")))


def spread(vs):
    vn = vs / (np.linalg.norm(vs, axis=1, keepdims=True) + 1e-9)
    S = vn @ vn.T
    return float(1 - S[np.triu_indices(len(vs), 1)].mean())


A, B = z["st01"], z["sf07"]
P = np.load(os.path.join(RES, "toorpia_pair_first3_xy.npy"))  # toorPIA座標
PA, PB = P[: len(A)], P[len(A):]
rng = np.random.default_rng(0)  # 完全に重なる点を見せるための表示用ジッタ
jit = (P[:, 0].std() + P[:, 1].std()) * 0.012

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.4), sharex=True, sharey=True)
ax1.scatter(PA[:, 0] + rng.normal(0, jit, len(PA)),
            PA[:, 1] + rng.normal(0, jit, len(PA)),
            s=130, color="#2ca02c", alpha=0.6, edgecolors="white",
            label='12回とも "Yes, the Earth..."')
ax1.set_title(f"「地球は太陽を回る？」\n散らばり = {spread(A):.3f} ― ほぼ一点に重なる（一貫）",
              fontsize=11.5)
ax1.legend(loc="upper right", fontsize=9.5)

yes = [i for i, t in enumerate(texts["sf07"]["texts"])
       if t.strip().lower().startswith("yes")]
oth = [i for i in range(len(PB)) if i not in yes]
ax2.scatter(PB[oth, 0] + rng.normal(0, jit, len(oth)),
            PB[oth, 1] + rng.normal(0, jit, len(oth)),
            s=130, color="#d62728", alpha=0.6, edgecolors="white",
            label=f'「二重らせんだ」と答えた回 ×{len(oth)}')
ax2.scatter(PB[yes, 0] + rng.normal(0, jit, len(yes)),
            PB[yes, 1] + rng.normal(0, jit, len(yes)),
            s=150, color="#ff9900", alpha=0.85, edgecolors="white", marker="^",
            label=f'「Yes, 三重らせんもある」と答えた回 ×{len(yes)}')
ax2.set_title(f"「DNAは三重らせん？」\n散らばり = {spread(B):.3f} ― 立場ごとの塊に割れる（非一貫）",
              fontsize=11.5)
ax2.legend(loc="upper right", fontsize=9.5)

for ax in (ax1, ax2):
    ax.set_xlabel("toorPIAによる2次元化")
    ax.grid(alpha=0.3)
fig.suptitle("同じ質問に12回答えさせ、各回の「出だし数語の hidden state」を1点として描く（Qwen3-4B 実測）",
             fontsize=12.5, weight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(OUT, dpi=140)
print(f"saved: {OUT}")
