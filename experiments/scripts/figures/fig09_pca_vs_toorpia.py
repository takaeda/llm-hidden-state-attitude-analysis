#!/usr/bin/env python3
"""教材図 images/09_pca_vs_toorpia.png を生成する（ADVANCED・2次元化手法の比較）。

同じ高次元データ（hidden state点群）を4通りの方法で2次元化して並べる:
  ① PCA・生のベクトル（ユークリッド幾何。PCA自体は平均中心化+分散最大方向への
     線形射影であり、規格化は行わない）
  ② PCA・各ベクトルをL2規格化してから（コサイン幾何相当）
  ③ toorPIA・規格化あり（vector_normalization=True, サーバー既定）
  ④ toorPIA・規格化なし（vector_normalization=False。ベクトル長＝迷いの強さの
     情報を保持。本編の図はこちらを採用）

上段: sf07「DNAは三重らせん？」12試行の出だし3語の状態
下段: 売上課題への14回のフル回答（全文平均）の状態

要点: 群構造（どの点が群れるか）は4方式とも一致。島どうしの距離の見た目は
方式に依存する。定量指標は2次元図ではなく高次元距離で直接計算している。

入力: results/Qwen3-4B/full_first3.npz, full_texts.json,
      complex_task.json, complex_task_vectors.npz,
      toorpia_pair_first3_xy.npy / _nonorm_xy.npy,
      toorpia_sales_xy.npy / _nonorm_xy.npy

usage: python fig09_pca_vs_toorpia.py
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

plt.rcParams["font.family"] = "IPAexGothic"


def make_square(ax, pts, pad=1.12):
    """縦横を同スケールにし、パネルを正方形にする"""
    x, y = pts[:, 0], pts[:, 1]
    cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    half = max(x.max() - x.min(), y.max() - y.min()) / 2 * pad + 1e-9
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")


EXP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(EXP)
RES = os.path.join(EXP, "results", "Qwen3-4B")
OUT = os.path.join(ROOT, "images", "09_pca_vs_toorpia.png")


def pca_raw(X):
    """PCAそのまま（中心化のみ。ユークリッド幾何）"""
    return PCA(n_components=2).fit_transform(X)


def pca_cos(X):
    """各行をL2規格化してからPCA（コサイン幾何相当）"""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return PCA(n_components=2).fit_transform(Xn)


# --- データ1: sf07 出だし3語（12点）---
z = np.load(os.path.join(RES, "full_first3.npz"))
texts = json.load(open(os.path.join(RES, "full_texts.json")))["sf07"]["texts"]
B = z["sf07"]
n_st = 12
pair = np.vstack([z["st01"], B])  # toorPIA/PCAとも st01 と同時に学習した座標
sf = {
    "PCA・生（ユークリッド）": pca_raw(pair)[n_st:],
    "PCA・L2規格化後（コサイン相当）": pca_cos(pair)[n_st:],
    "toorPIA・規格化あり": np.load(os.path.join(RES, "toorpia_pair_first3_xy.npy"))[n_st:],
    "toorPIA・規格化なし": np.load(os.path.join(RES, "toorpia_pair_first3_nonorm_xy.npy"))[n_st:],
}
yes = [i for i, t in enumerate(texts) if t.strip().lower().startswith("yes")]
dna_t = [i for i in range(len(B)) if i not in yes and not texts[i].startswith("DNA (")]
dna_p = [i for i in range(len(B)) if i not in yes and texts[i].startswith("DNA (")]

# --- データ2: sales 全文平均（14点）---
d = json.load(open(os.path.join(RES, "complex_task.json")))
sa_texts = d["samples"]["complex_sales"]
V = np.load(os.path.join(RES, "complex_task_vectors.npz"))["complex_sales"]
sa = {
    "PCA・生（ユークリッド）": pca_raw(V),
    "PCA・L2規格化後（コサイン相当）": pca_cos(V),
    "toorPIA・規格化あり": np.load(os.path.join(RES, "toorpia_sales_xy.npy")),
    "toorPIA・規格化なし": np.load(os.path.join(RES, "toorpia_sales_nonorm_xy.npy")),
}
cust = [i for i, t in enumerate(sa_texts) if "egment" in t]
deco = [i for i, t in enumerate(sa_texts) if "decomposition" in t.lower()]
ana = [i for i in range(len(V)) if i not in cust and i not in deco]

fig, axes = plt.subplots(2, 4, figsize=(18, 10.2))
rng = np.random.default_rng(0)


def draw_sf(ax, P, title):
    jit = (P[:, 0].std() + P[:, 1].std()) * 0.012
    J = lambda idx: (P[idx, 0] + rng.normal(0, jit, len(idx)),
                     P[idx, 1] + rng.normal(0, jit, len(idx)))
    ax.scatter(*J(dna_t), s=110, color="#d62728", alpha=0.7, edgecolors="white",
               label=f'二重らせん派 typically系 ×{len(dna_t)}（exists×5/has×2の2小島）')
    ax.scatter(*J(dna_p), s=110, facecolors="none", edgecolors="#d62728", lw=2,
               label=f'二重らせん派 "DNA (正式名称)…" ×{len(dna_p)}')
    ax.scatter(*J(yes), s=140, color="#ff9900", marker="^", alpha=0.9,
               edgecolors="white", label=f"三重らせん派 ×{len(yes)}")
    ax.set_title(title, fontsize=10.5, weight="bold")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.3)
    make_square(ax, P)


def draw_sa(ax, P, title):
    ax.scatter(P[ana, 0], P[ana, 1], s=110, color="#d62728", alpha=0.7,
               edgecolors="white", label=f"時系列の分析 ×{len(ana)}")
    ax.scatter(P[deco, 0], P[deco, 1], s=110, facecolors="none",
               edgecolors="#d62728", lw=2, label=f"時系列の分解 ×{len(deco)}")
    ax.scatter(P[cust, 0], P[cust, 1], s=140, color="#1f77b4", marker="^",
               alpha=0.9, edgecolors="white", label=f"顧客分析 ×{len(cust)}")
    ax.set_title(title, fontsize=10.5, weight="bold")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.3)
    make_square(ax, P)


for j, (name, P) in enumerate(sf.items()):
    draw_sf(axes[0, j], P, name + "\nsf07 出だし3語（12点）")
axes[0, 0].legend(fontsize=7.5)
for j, (name, P) in enumerate(sa.items()):
    draw_sa(axes[1, j], P, name + "\n売上課題・全文（14点）")
axes[1, 0].legend(fontsize=7.5)

fig.suptitle("参考: 同じ高次元データを4通りの方法で2次元化（Qwen3-4B）\n"
             "群構造（どの点が群れるか）は4方式とも一致。島どうしの距離の見た目は方式に依存する。\n"
             "本編の図は「toorPIA・規格化なし」（ベクトル長＝迷いの強さの情報を保持）を採用 ― 定量指標は高次元距離で直接計算",
             fontsize=12.5, weight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(OUT, dpi=130)
print(f"saved: {OUT}")
