#!/usr/bin/env python3
"""dispersion_vectors.csv（240試行 × 2560次元 hidden state）を toorPIA に投入し、
ベースマップ上の2D座標を取得 → 質問ごとのクラスタの広がりを測り、
確信度（行動の迷い）と相関するかを検証する。

「全次元の広がりは確信度を見ない／PCA低次元では見える」という前回結果を、
toorPIA の非線形ベースマップで測り直す。
"""
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from toorpia.client import toorPIA

plt.rcParams["font.family"] = "IPAexGothic"
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(BASE, "results", "Qwen3-4B", "dispersion_vectors.csv")

rows = list(csv.DictReader(open(CSV)))
qids = [r["qid"] for r in rows]
cats = [r["category"] for r in rows]

client = toorPIA()
print("creating toorPIA basemap from 2560-dim hidden states (240 points)...")
xy = client.fit_transform_csvform(
    CSV,
    drop_columns=["qid", "category", "trial"],
    label="LLM hidden-state dispersion (Qwen3-4B)",
    tag="attitude-probe",
    description="20 trials x 12 questions; mean-pooled last-layer hidden state",
)
xy = np.asarray(xy, dtype=float)
print("coords shape:", xy.shape, "| shareUrl:", getattr(client, "shareUrl", None))
np.save(os.path.join(BASE, "results", "Qwen3-4B", "toorpia_xy.npy"), xy)

disp = json.load(open(os.path.join(BASE, "results", "Qwen3-4B", "dispersion.json")))
pq = disp["per_question"]

uq = list(dict.fromkeys(qids))
spread_toor, beh_h = {}, {}
for q in uq:
    idx = [i for i, x in enumerate(qids) if x == q]
    p = xy[idx]
    spread_toor[q] = float(np.mean(np.linalg.norm(p - p.mean(0), axis=1)))
    beh_h[q] = pq[q]["behavior_entropy"]

cats4 = ["science_true", "science_false", "contested", "political"]
catja = {"science_true": "科学・正", "science_false": "科学・偽",
         "contested": "議論", "political": "政治"}
catcol = {"science_true": "#2ca02c", "science_false": "#d62728",
          "contested": "#ff7f0e", "political": "#9467bd"}

# 相関
xs = [spread_toor[q] for q in uq]
ys = [beh_h[q] for q in uq]
r = float(np.corrcoef(xs, ys)[0, 1])
by_cat = {c: float(np.mean([spread_toor[q] for q in uq
          if cats[[i for i, x in enumerate(qids) if x == q][0]] == c]))
          for c in cats4}
sci = np.mean([by_cat["science_true"], by_cat["science_false"]])
unc = np.mean([by_cat["contested"], by_cat["political"]])

out = {"spread_toorpia_by_question": spread_toor,
       "spread_by_category": by_cat,
       "pearson_spread_vs_behavior_entropy": round(r, 3),
       "science_mean": round(sci, 4), "uncertain_mean": round(unc, 4),
       "shareUrl": getattr(client, "shareUrl", None)}
json.dump(out, open(os.path.join(BASE, "results", "Qwen3-4B",
          "toorpia_spread.json"), "w"), ensure_ascii=False, indent=1)

# 図1: toorPIAベースマップ（カテゴリ色分け）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
for q in uq:
    idx = [i for i, x in enumerate(qids) if x == q]
    c = catcol[cats[idx[0]]]
    ax1.scatter(xy[idx, 0], xy[idx, 1], s=45, color=c, alpha=0.7, edgecolors="white")
    cx, cy = xy[idx, 0].mean(), xy[idx, 1].mean()
    ax1.annotate(q, (cx, cy), fontsize=9, weight="bold")
ax1.set_title("toorPIA ベースマップ：各質問20試行の hidden state\n（クラスタの広がり=その質問への内部状態のばらつき）", fontsize=11)
ax1.set_xlabel("toorPIA-x"); ax1.set_ylabel("toorPIA-y")
from matplotlib.lines import Line2D
ax1.legend(handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor=catcol[k],
           markersize=9, label=catja[k]) for k in cats4], fontsize=9)
ax1.grid(alpha=0.3)

# 図2: 広がり vs 行動の迷い
for q in uq:
    ax2.scatter(spread_toor[q], beh_h[q], s=90,
                color=catcol[cats[[i for i, x in enumerate(qids) if x == q][0]]],
                edgecolors="white", zorder=3)
    ax2.annotate(q, (spread_toor[q], beh_h[q]), fontsize=8,
                 xytext=(4, 4), textcoords="offset points")
ax2.set_xlabel("toorPIAマップ上のクラスタの広がり")
ax2.set_ylabel("行動の迷い（回答の割れ方エントロピー）")
ax2.set_title(f"広がり ↔ 確信度（toorPIA空間）\nPearson r = {r:.3f}  /  科学={sci:.3f} < 政治議論={unc:.3f}", fontsize=11)
ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(BASE, "results", "Qwen3-4B", "toorpia_dispersion.png"), dpi=140)
print(f"\nPearson(spread, behavior entropy) = {r:.3f}")
print(f"by category: {by_cat}")
print(f"science={sci:.4f}  uncertain(political+contested)={unc:.4f}")
print("saved toorpia_dispersion.png, toorpia_spread.json")
