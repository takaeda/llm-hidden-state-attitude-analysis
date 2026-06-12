#!/usr/bin/env python3
"""本編図 images/e02_two_questions.png を生成する（平易版・2問の点群）。

「頭の中を点にすると、迷いのない質問は1点に重なり、迷う質問は割れる」
だけを伝える。2次元化は toorPIA（本講義の道具）。

入力: results/Qwen3-4B/full_first3.npz, full_texts.json,
      toorpia_pair_first3_xy.npy（toorPIAが返した2D座標キャッシュ。
        無ければ toorPIA API に投入して生成。同一行が多いため微小ジッタを
        加えて投入する。要 TOORPIA_API_KEY/URL）

usage: python fig_e02_two_questions.py
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
OUT = os.path.join(ROOT, "images", "e02_two_questions.png")
XY = os.path.join(RES, "toorpia_pair_first3_xy.npy")

z = np.load(os.path.join(RES, "full_first3.npz"))
texts = json.load(open(os.path.join(RES, "full_texts.json")))
A, B = z["st01"], z["sf07"]

if os.path.exists(XY):
    P = np.load(XY)
else:  # toorPIA API へ投入（要APIキー）
    from toorpia.client import toorPIA
    X = np.vstack([A, B])
    rng = np.random.default_rng(42)
    Xj = X + rng.normal(0, 1, X.shape) * (X.std(0, keepdims=True) * 1e-3 + 1e-8)
    tmp = "/tmp/pair-first3.csv"
    ids = ["st01"] * len(A) + ["sf07"] * len(B)
    with open(tmp, "w") as f:
        f.write("qid,trial," + ",".join(f"d{i}" for i in range(X.shape[1])) + "\n")
        for k, row in enumerate(Xj):
            f.write(f"{ids[k]},{k}," + ",".join(f"{v:.5f}" for v in row) + "\n")
    client = toorPIA()
    P = np.asarray(client.fit_transform_csvform(
        tmp, drop_columns=["qid", "trial"], label="pair-first3"), dtype=float)
    np.save(XY, P)

PA, PB = P[: len(A)], P[len(A):]
rng = np.random.default_rng(0)  # 完全に重なる点を見せるための表示用ジッタ
jit = (P[:, 0].std() + P[:, 1].std()) * 0.012

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.6), sharex=True, sharey=True)
ax1.scatter(PA[:, 0] + rng.normal(0, jit, len(PA)),
            PA[:, 1] + rng.normal(0, jit, len(PA)),
            s=200, color="#2ca02c", alpha=0.6, edgecolors="white")
ax1.set_title("「地球は太陽を回る？」×12回", fontsize=15, weight="bold")
ax1.text(0.5, 0.13, "12個の点が、ぴったり重なっている\n＝ 迷いなし",
         transform=ax1.transAxes, ha="center", fontsize=13, color="#2a7",
         weight="bold",
         bbox=dict(boxstyle="round,pad=0.4", fc="#eaffea", ec="#7a7"))

sf_texts = texts["sf07"]["texts"]
yes = [i for i, t in enumerate(sf_texts) if t.strip().lower().startswith("yes")]
oth = [i for i in range(len(PB)) if i not in yes]
ax2.scatter(PB[oth, 0] + rng.normal(0, jit, len(oth)),
            PB[oth, 1] + rng.normal(0, jit, len(oth)),
            s=200, color="#d62728", alpha=0.6, edgecolors="white",
            label='「二重らせんです」派')
ax2.scatter(PB[yes, 0] + rng.normal(0, jit, len(yes)),
            PB[yes, 1] + rng.normal(0, jit, len(yes)),
            s=230, color="#ff9900", alpha=0.85, edgecolors="white", marker="^",
            label='「はい、三重らせんもあります」派')
ax2.set_title("「DNAは三重らせん？」×12回", fontsize=15, weight="bold")
ax2.legend(fontsize=12, loc="center left")
ax2.text(0.97, 0.96, "点がグループに割れている\n＝ 答えがゆれている",
         transform=ax2.transAxes, ha="right", va="top", fontsize=13,
         color="#a33", weight="bold",
         bbox=dict(boxstyle="round,pad=0.4", fc="#ffecec", ec="#c88"))

# 赤の2つの島の正体（言い回し違い・中身は同じ）を注記
top = [i for i in oth if not sf_texts[i].startswith("DNA (")]
bot = [i for i in oth if sf_texts[i].startswith("DNA (")]
ax2.annotate(f'"DNA typically …" と始めた回 ×{len(top)}',
             xy=(PB[top, 0].mean(), PB[top, 1].mean()),
             xytext=(20, -45), textcoords="offset points",
             fontsize=10.5, color="#a33",
             arrowprops=dict(arrowstyle="->", color="#a33"))
ax2.annotate(f'"DNA (正式名称) …" と始めた回 ×{len(bot)}\n'
             "（中身はどちらも「二重らせん」。\n  言い回しが違うだけで島が分かれる）",
             xy=(PB[bot, 0].mean(), PB[bot, 1].mean()),
             xytext=(25, 35), textcoords="offset points",
             fontsize=10.5, color="#a33",
             arrowprops=dict(arrowstyle="->", color="#a33"))

for ax in (ax1, ax2):
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("同じ質問に12回答えさせて、そのときの「頭の中」を1点ずつ打ってみた"
             "（toorPIAによる2次元化／使用モデル: Qwen3-4B）",
             fontsize=14, weight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(OUT, dpi=140)
print(f"saved: {OUT}")
