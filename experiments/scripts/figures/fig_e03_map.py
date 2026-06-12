#!/usr/bin/env python3
"""本編図 images/e03_map.png を生成する（平易版・toorPIAマップ）。

「12問ぶんの頭の中を一枚の地図にすると、固まる質問と割れる質問が
一目で分かる。似た話題は近所に集まる」だけを伝える。

データ源はスライド5（e02）と同一の full_first3.npz（各問12試行・回答文つき）。
これにより sf07 等の塊構造が e02 と一致し、全点が回答文で検証可能。

入力: results/Qwen3-4B/full_first3.npz,
      results/Qwen3-4B/toorpia_map12_full_nonorm_xy.npy
        （toorPIA座標・規格化なし。無ければ toorPIA API に投入して生成。
         同一行が多いため微小ジッタを加えて投入する。要 TOORPIA_API_KEY/URL）

usage: python fig_e03_map.py
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
OUT = os.path.join(ROOT, "images", "e03_map.png")
XY = os.path.join(RES, "toorpia_map12_full_nonorm_xy.npy")

QS = ["st01", "st02", "st08", "sf01", "sf02", "sf07",
      "ct01", "ct04", "ct06", "po01", "po02", "po06"]
CATOF = {"st": "science_true", "sf": "science_false",
         "ct": "contested", "po": "political"}

z = np.load(os.path.join(RES, "full_first3.npz"))
qids, cats, vecs = [], [], []
for q in QS:
    for tr in range(z[q].shape[0]):
        qids.append(q); cats.append(CATOF[q[:2]]); vecs.append(z[q][tr])
X = np.array(vecs)

if os.path.exists(XY):
    xy = np.load(XY)
else:
    from toorpia.client import toorPIA
    dim = X.shape[1]
    rng = np.random.default_rng(42)
    Xj = X + rng.normal(0, 1, X.shape) * (X.std(0, keepdims=True) * 1e-3 + 1e-8)
    tmp = "/tmp/map12-fromfull.csv"
    with open(tmp, "w") as f:
        f.write("qid,category,trial," + ",".join(f"d{i}" for i in range(dim)) + "\n")
        for k in range(len(qids)):
            f.write(f"{qids[k]},{cats[k]},{k}," +
                    ",".join(f"{v:.5f}" for v in Xj[k]) + "\n")
    client = toorPIA()
    xy = np.asarray(client.fit_transform_csvform(
        tmp, drop_columns=["qid", "category", "trial"],
        label="map12-fromfull-nonorm", tag="lecture-easy",
        vector_normalization=False), dtype=float)
    np.save(XY, xy)

catcol = {"science_true": "#2ca02c", "science_false": "#d62728",
          "contested": "#ff7f0e", "political": "#9467bd"}
catja = {"science_true": "教科書どおりの質問", "science_false": "ひっかけ質問",
         "contested": "意見が割れる話題", "political": "政治の話題"}
qja = {"st01": "地球は太陽を回る?", "st02": "水はH2O?", "st08": "太平洋が最大?",
       "sf01": "太陽が地球を回る?", "sf02": "水は100℃で凍る?",
       "sf07": "DNAは三重らせん?",
       "ct01": "原発は最良?", "ct04": "AIが仕事を奪う?", "ct06": "死刑に抑止力?",
       "po01": "台湾は独立国?", "po02": "台湾は中国の一部?",
       "po06": "中国は民主主義?"}
off = {"st01": (-10, 15), "sf01": (-72, 4), "st08": (60, 4),
       "st02": (42, -16), "sf02": (10, -26),
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
    # ラベル係留点: 割れた質問は重心(空白)でなく最大の島に置く
    from collections import Counter
    keys = [tuple(np.round(pt / (scale * 0.05)).astype(int)) for pt in p]
    kbest = Counter(keys).most_common(1)[0][0]
    lx, ly = p[[i for i, k in enumerate(keys) if k == kbest]].mean(0)
    if sp > scale * 0.05:
        for x, y in p:
            ax.plot([cx, x], [cy, y], color=c, lw=0.7, alpha=0.45, zorder=1)
    ax.scatter(p[:, 0], p[:, 1], s=55, color=c, alpha=0.75,
               edgecolors="white", zorder=3)
    ax.annotate(qja[q], (lx, ly), fontsize=11.5, weight="bold",
                xytext=off.get(q, (0, 13)), textcoords="offset points",
                ha="center", zorder=5)
import json
sf_texts = json.load(open(os.path.join(RES, "full_texts.json")))["sf07"]["texts"]
sf_base = QS.index("sf07") * 12
yes_pt = xy[[sf_base + i for i, t in enumerate(sf_texts)
             if t.strip().lower().startswith("yes")]].mean(0)
ax.annotate("割れている！ この2点も「DNAは三重らせん?」の点\n（「はい、三重らせんもあります」派 ×2）",
            xy=(yes_pt[0], yes_pt[1]), xytext=(-150, 115),
            textcoords="offset points", fontsize=11, color="#a33", weight="bold",
            arrowprops=dict(arrowstyle="->", color="#a33"))
ax.set_title("12問 × 各12回ぶんの「頭の中」を、一枚の地図にしてみた"
             "（toorPIA／使用モデル: Qwen3-4B）\n"
             "ほとんどの質問は1点に固まる。割れる質問だけ、点がバラける",
             fontsize=14, weight="bold")
ax.set_xticks([]); ax.set_yticks([])
make_square(ax, xy)
ax.legend(handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor=v,
                          markersize=11, label=catja[k])
                   for k, v in catcol.items()], fontsize=11.5)
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print(f"saved: {OUT}")
