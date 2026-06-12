#!/usr/bin/env python3
"""本編図 images/e02_two_questions.png を生成する（平易版・2問の点群）。

「頭の中を点にすると、迷いのない質問は1点に重なり、迷う質問は割れる」を伝える。
2次元化は toorPIA（規格化なし）。座標はスライド6の地図(toorpia_map12_full_
nonorm_xy.npy, 12問×各12試行=144点で学習)と同一マップから切り出す
＝スライド5とスライド6で点の配置が完全に一致する。

sf07 の事実（全点をベクトル・回答文と突き合わせて検証済み）:
  出だし3語の状態は4種類 — Yes三重×2 / "DNA typically exists"×5 /
  "DNA typically has"×2 / "DNA (正式名称)"×3。
  二重らせん派(赤)は言い回しごとに3つの小島になる。

入力: results/Qwen3-4B/full_first3.npz, full_texts.json,
      toorpia_map12_full_nonorm_xy.npy（無ければ先に fig_e03_map.py を実行）

usage: python fig_e02_two_questions.py
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "IPAexGothic"


def make_square(ax, pts, pad=1.30):
    x, y = pts[:, 0], pts[:, 1]
    cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    half = max(x.max() - x.min(), y.max() - y.min()) / 2 * pad + 1e-9
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")


EXP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(EXP)
RES = os.path.join(EXP, "results", "Qwen3-4B")
OUT = os.path.join(ROOT, "images", "e02_two_questions.png")
XY = os.path.join(RES, "toorpia_map12_full_nonorm_xy.npy")

QS = ["st01", "st02", "st08", "sf01", "sf02", "sf07",
      "ct01", "ct04", "ct06", "po01", "po02", "po06"]
if not os.path.exists(XY):
    raise SystemExit("座標キャッシュが無い。先に fig_e03_map.py を実行すること")
xy = np.load(XY)
P_st = xy[QS.index("st01") * 12:QS.index("st01") * 12 + 12]
P_sf = xy[QS.index("sf07") * 12:QS.index("sf07") * 12 + 12]

texts = json.load(open(os.path.join(RES, "full_texts.json")))["sf07"]["texts"]
yes = [i for i, t in enumerate(texts) if t.strip().lower().startswith("yes")]
g_exists = [i for i, t in enumerate(texts) if t.startswith("DNA typically exists")]
g_has = [i for i, t in enumerate(texts) if t.startswith("DNA typically has")]
g_paren = [i for i, t in enumerate(texts) if t.startswith("DNA (")]
assert sorted(yes + g_exists + g_has + g_paren) == list(range(12))

rng = np.random.default_rng(0)  # 完全に重なる点を見せるための表示用ジッタ
jit = (np.vstack([P_st, P_sf]).std()) * 0.02

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 6.8),
                               sharex=True, sharey=True)
J = lambda Q, idx: (Q[idx, 0] + rng.normal(0, jit, len(idx)),
                    Q[idx, 1] + rng.normal(0, jit, len(idx)))

ax1.scatter(*J(P_st, list(range(12))), s=200, color="#2ca02c", alpha=0.6,
            edgecolors="white")
ax1.set_title("「地球は太陽を回る？」×12回", fontsize=15, weight="bold")
ax1.text(0.5, 0.10, "12個の点が、ぴったり重なっている\n＝ 迷いなし",
         transform=ax1.transAxes, ha="center", fontsize=13, color="#2a7",
         weight="bold",
         bbox=dict(boxstyle="round,pad=0.4", fc="#eaffea", ec="#7a7"))

ax2.scatter(*J(P_sf, g_exists + g_has + g_paren), s=200, color="#d62728",
            alpha=0.6, edgecolors="white", label="「二重らせんです」派 ×10")
ax2.scatter(*J(P_sf, yes), s=230, color="#ff9900", alpha=0.85,
            edgecolors="white", marker="^",
            label="「はい、三重らせんもあります」派 ×2")
ax2.set_title("「DNAは三重らせん？」×12回", fontsize=15, weight="bold")
ax2.legend(fontsize=11.5, loc="upper left")
ax2.text(0.97, 0.80, "点がグループに割れている\n＝ 答えがゆれている",
         transform=ax2.transAxes, ha="right", va="top", fontsize=13,
         color="#a33", weight="bold",
         bbox=dict(boxstyle="round,pad=0.4", fc="#ffecec", ec="#c88"))

# 赤の3つの小島の正体（言い回し違い・中身は同じ）を注記
for idx, lab, dxy in [
        (g_exists, f'"DNA typically exists…" ×{len(g_exists)}', (-55, 38)),
        (g_has, f'"DNA typically has…" ×{len(g_has)}', (30, 65)),
        (g_paren, f'"DNA (正式名称) …" ×{len(g_paren)}', (70, 30))]:
    ax2.annotate(lab, xy=(P_sf[idx, 0].mean(), P_sf[idx, 1].mean()),
                 xytext=dxy, textcoords="offset points",
                 fontsize=10, color="#a33", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#a33"))
ax2.text(0.03, 0.40, "赤の3つの小島＝言い回しが3通りあるだけ\n（中身はぜんぶ「二重らせん」）",
         transform=ax2.transAxes, fontsize=10.5, color="#a33",
         bbox=dict(boxstyle="round,pad=0.3", fc="#fff5f5", ec="#daa"))

for ax in (ax1, ax2):
    ax.set_xticks([]); ax.set_yticks([])
    make_square(ax, np.vstack([P_st, P_sf]))
fig.suptitle("同じ質問に12回答えさせて、そのときの「頭の中」を1点ずつ打ってみた"
             "（toorPIAによる2次元化／使用モデル: Qwen3-4B）",
             fontsize=14, weight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(OUT, dpi=140)
print(f"saved: {OUT}")
