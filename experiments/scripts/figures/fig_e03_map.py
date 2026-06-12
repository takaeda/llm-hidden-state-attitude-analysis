#!/usr/bin/env python3
"""本編図 images/e03_map.png を生成する（平易版・toorPIAマップ）。

「12問ぶんの頭の中を一枚の地図にすると、固まる質問と割れる質問が
一目で分かる。似た話題は近所に集まる」だけを伝える。

入力: results/Qwen3-4B/first3_vectors.csv, toorpia_first3_xy.npy

usage: python fig_e03_map.py
"""
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "IPAexGothic"
EXP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(EXP)
RES = os.path.join(EXP, "results", "Qwen3-4B")
OUT = os.path.join(ROOT, "images", "e03_map.png")

rows = list(csv.DictReader(open(os.path.join(RES, "first3_vectors.csv"))))
qids = [r["qid"] for r in rows]
cats = [r["category"] for r in rows]
xy = np.load(os.path.join(RES, "toorpia_first3_xy.npy"))

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
off = {"sf07": (0, 16), "po01": (52, 8), "po02": (-58, 8)}

uq = list(dict.fromkeys(qids))
fig, ax = plt.subplots(figsize=(11, 8))
for q in uq:
    idx = [i for i, x in enumerate(qids) if x == q]
    p = xy[idx]
    c = catcol[cats[idx[0]]]
    cx, cy = p[:, 0].mean(), p[:, 1].mean()
    sp = float(np.mean(np.linalg.norm(p - p.mean(0), axis=1)))
    if sp > 0.01:
        for x, y in p:
            ax.plot([cx, x], [cy, y], color=c, lw=0.8, alpha=0.45, zorder=1)
    ax.scatter(p[:, 0], p[:, 1], s=60, color=c, alpha=0.75,
               edgecolors="white", zorder=3)
    ax.annotate(qja[q], (cx, cy), fontsize=11.5, weight="bold",
                xytext=off.get(q, (0, 13)), textcoords="offset points",
                ha="center", zorder=5)
ax.annotate("割れている！", xy=(-0.22, 0.30), xytext=(-0.13, 0.16),
            fontsize=13, color="#a33", weight="bold",
            arrowprops=dict(arrowstyle="->", color="#a33"))
ax.set_title("12問 × 20回ぶんの「頭の中」を、一枚の地図にしてみた（toorPIA）\n"
             "ほとんどの質問は1点に固まる。割れる質問だけ、点がバラける",
             fontsize=14, weight="bold")
ax.set_xticks([]); ax.set_yticks([])
ax.legend(handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor=v,
                          markersize=11, label=catja[k])
                   for k, v in catcol.items()], fontsize=11.5)
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print(f"saved: {OUT}")
