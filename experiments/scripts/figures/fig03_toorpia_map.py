#!/usr/bin/env python3
"""教材図 images/03_toorpia_map.png を生成する（ADVANCED スライド8）。

12問 × 各12試行の出だし3トークンhidden state を toorPIA で2D化（規格化なし）。
質問ごとの点群（密集=一貫 / 複数の塊=割れ）を描き、割れた質問はスパイダー線で
所属を明示する。データ源は本編 e03 と同一（full_first3.npz）。

入力: results/Qwen3-4B/full_first3.npz,
      results/Qwen3-4B/toorpia_map12_full_nonorm_xy.npy
        （無ければ先に fig_e03_map.py を実行して座標を生成。要 API）

usage: python fig03_toorpia_map.py
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "IPAexGothic"


def make_square(ax, pts, pad=1.12):
    x, y = pts[:, 0], pts[:, 1]
    cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    half = max(x.max() - x.min(), y.max() - y.min()) / 2 * pad + 1e-9
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")


EXP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(EXP)
RES = os.path.join(EXP, "results", "Qwen3-4B")
OUT = os.path.join(ROOT, "images", "03_toorpia_map.png")
XY = os.path.join(RES, "toorpia_map12_full_nonorm_xy.npy")

QS = ["st01", "st02", "st08", "sf01", "sf02", "sf07",
      "ct01", "ct04", "ct06", "po01", "po02", "po06"]
CATOF = {"st": "science_true", "sf": "science_false",
         "ct": "contested", "po": "political"}

z = np.load(os.path.join(RES, "full_first3.npz"))
qids, cats = [], []
for q in QS:
    for tr in range(z[q].shape[0]):
        qids.append(q); cats.append(CATOF[q[:2]])

if not os.path.exists(XY):
    raise SystemExit("座標キャッシュが無い。先に fig_e03_map.py を実行すること")
xy = np.load(XY)

catcol = {"science_true": "#2ca02c", "science_false": "#d62728",
          "contested": "#ff7f0e", "political": "#9467bd"}
catja = {"science_true": "科学・正しい命題", "science_false": "科学・誤った命題",
         "contested": "議論が分かれる", "political": "政治・センシティブ"}
qja = {"st01": "地球→太陽", "st02": "水=H2O", "st08": "太平洋最大",
       "sf01": "太陽→地球?", "sf02": "水100℃で凍る?", "sf07": "DNA三重らせん?",
       "ct01": "原発は最良?", "ct04": "AIが仕事を奪う?", "ct06": "死刑に抑止力?",
       "po01": "台湾は独立国?", "po02": "台湾は中国の一部?", "po06": "中国は民主主義?"}
off = {"st01": (-10, 15), "sf01": (-72, 4), "st08": (60, 4),
       "st02": (42, -16), "sf02": (-54, -16),
       "ct01": (0, 16), "ct06": (54, 6), "ct04": (12, -20),
       "po01": (56, 7), "po02": (-64, -9), "po06": (0, 16),
       "sf07": (0, 16)}

scale = (xy[:, 0].std() + xy[:, 1].std())
fig, ax = plt.subplots(figsize=(10, 10.4))
for q in QS:
    idx = [i for i, x in enumerate(qids) if x == q]
    p = xy[idx]
    c = catcol[cats[idx[0]]]
    cx, cy = p[:, 0].mean(), p[:, 1].mean()
    sp = float(np.mean(np.linalg.norm(p - p.mean(0), axis=1)))
    if sp > scale * 0.05:
        for x, y in p:
            ax.plot([cx, x], [cy, y], color=c, lw=0.7, alpha=0.45, zorder=1)
    ax.scatter(p[:, 0], p[:, 1], s=55, color=c, alpha=0.7,
               edgecolors="white", zorder=3)
    ax.annotate(qja[q], (cx, cy), fontsize=9.5, weight="bold",
                xytext=off.get(q, (0, 11)), textcoords="offset points",
                ha="center", zorder=5)
ax.set_title("toorPIA マップ（規格化なし）：12問 × 各12回の「出だし数語の hidden state」"
             "（Qwen3-4B 実測）\n"
             "ほとんどの質問は12回ぶんが一点に固まる（一貫）。割れる質問だけが複数の塊に分かれる（線=同じ質問）",
             fontsize=11)
ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.3)
make_square(ax, xy)
ax.legend(handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor=v,
                          markersize=9, label=catja[k]) for k, v in catcol.items()],
          fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print(f"saved: {OUT}")
