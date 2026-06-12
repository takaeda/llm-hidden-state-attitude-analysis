#!/usr/bin/env python3
"""教材図 images/09_pca_vs_toorpia.png を生成する（ADVANCED・PCAとtoorPIAの比較）。

同じ高次元データ（hidden state点群）を PCA と toorPIA で2次元化して並べる。
上段: sf07「DNAは三重らせん？」12試行の出だし3語の状態
下段: 売上課題への14回のフル回答（全文平均）の状態
色=立場、▲=例外の立場、塗り/白抜き=同一立場内の言い回し・粒度の違い。

要点: 2次元図の見た目（島どうしの距離の比率）は手法で変わる。
本資料の定量指標（散らばり・分離度）は2次元図ではなく高次元での
コサイン距離で計算しており、図の見た目には依存しない。

入力: results/Qwen3-4B/full_first3.npz, full_texts.json,
      complex_task.json, complex_task_vectors.npz,
      toorpia_pair_first3_xy.npy, toorpia_sales_xy.npy

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


def pca2(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return PCA(n_components=2).fit_transform(Xn)


# --- データ1: sf07 出だし3語（12点）---
z = np.load(os.path.join(RES, "full_first3.npz"))
texts = json.load(open(os.path.join(RES, "full_texts.json")))["sf07"]["texts"]
B = z["sf07"]
n_st = 12
P_pair = np.load(os.path.join(RES, "toorpia_pair_first3_xy.npy"))
sf_toor = P_pair[n_st:]                    # toorPIA（st01と同時投入したもの）
sf_pca = pca2(np.vstack([z["st01"], B]))[n_st:]  # PCAも同条件（24点で学習）
yes = [i for i, t in enumerate(texts) if t.strip().lower().startswith("yes")]
dna_t = [i for i in range(len(B)) if i not in yes and not texts[i].startswith("DNA (")]
dna_p = [i for i in range(len(B)) if i not in yes and texts[i].startswith("DNA (")]

# --- データ2: sales 全文平均（14点）---
d = json.load(open(os.path.join(RES, "complex_task.json")))
sa_texts = d["samples"]["complex_sales"]
V = np.load(os.path.join(RES, "complex_task_vectors.npz"))["complex_sales"]
sa_toor = np.load(os.path.join(RES, "toorpia_sales_xy.npy"))
sa_pca = pca2(V)
cust = [i for i, t in enumerate(sa_texts) if "egment" in t]
deco = [i for i, t in enumerate(sa_texts) if "decomposition" in t.lower()]
ana = [i for i in range(len(V)) if i not in cust and i not in deco]

fig, axes = plt.subplots(2, 2, figsize=(11.5, 12.2))
rng = np.random.default_rng(0)


def draw_sf(ax, P, title):
    jit = (P[:, 0].std() + P[:, 1].std()) * 0.012
    J = lambda idx: (P[idx, 0] + rng.normal(0, jit, len(idx)),
                     P[idx, 1] + rng.normal(0, jit, len(idx)))
    ax.scatter(*J(dna_t), s=120, color="#d62728", alpha=0.7, edgecolors="white",
               label=f'二重らせん派 "DNA typically…" ×{len(dna_t)}')
    ax.scatter(*J(dna_p), s=120, facecolors="none", edgecolors="#d62728", lw=2,
               label=f'二重らせん派 "DNA (正式名称)…" ×{len(dna_p)}')
    ax.scatter(*J(yes), s=150, color="#ff9900", marker="^", alpha=0.9,
               edgecolors="white", label=f"三重らせん派 ×{len(yes)}")
    ax.set_title(title, fontsize=11.5, weight="bold")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.3)
    make_square(ax, P)


def draw_sa(ax, P, title):
    ax.scatter(P[ana, 0], P[ana, 1], s=120, color="#d62728", alpha=0.7,
               edgecolors="white", label=f"時系列の分析 ×{len(ana)}")
    ax.scatter(P[deco, 0], P[deco, 1], s=120, facecolors="none",
               edgecolors="#d62728", lw=2, label=f"時系列の分解 ×{len(deco)}")
    ax.scatter(P[cust, 0], P[cust, 1], s=150, color="#1f77b4", marker="^",
               alpha=0.9, edgecolors="white", label=f"顧客分析 ×{len(cust)}")
    ax.set_title(title, fontsize=11.5, weight="bold")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.3)
    make_square(ax, P)


draw_sf(axes[0, 0], sf_pca, "PCA ― sf07 出だし3語の状態（12点）")
draw_sf(axes[0, 1], sf_toor, "toorPIA ― 同じデータ")
axes[0, 0].legend(fontsize=8.5)
draw_sa(axes[1, 0], sa_pca, "PCA ― 売上課題・全文の状態（14点）")
draw_sa(axes[1, 1], sa_toor, "toorPIA ― 同じデータ")
axes[1, 0].legend(fontsize=8.5)

fig.suptitle("参考: 同じ高次元データを PCA と toorPIA で2次元化して比較（Qwen3-4B）\n"
             "グループ構造（どの点が群れるか）は両者で一致。島どうしの距離の見た目は手法に依存する\n"
             "― 本資料の定量指標は2次元図ではなく高次元のコサイン距離で計算（図の見た目に依存しない） ―",
             fontsize=12, weight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(OUT, dpi=140)
print(f"saved: {OUT}")
