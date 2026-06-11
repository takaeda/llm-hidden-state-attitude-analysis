#!/usr/bin/env python3
"""教材図 images/03_toorpia_map.png を生成する（スライド8）。

12問 × 各20試行の出だし3トークンhidden state を toorPIA ベースマップで2D化し、
質問ごとの点群（密集=一貫 / 複数の塊=割れ）を描く。割れた質問は重心から
各点へ細線（スパイダー線）を引いて所属を明示する。

入力: results/Qwen3-4B/first3_vectors.csv（06_window_spread.py の出力）
      results/Qwen3-4B/toorpia_first3_xy.npy（toorPIAが返した2D座標。
        無ければ toorPIA API に投入して生成する。要 TOORPIA_API_KEY/URL。
        完全に同一の行が多いとAPIが弾くため、微小ジッタを加えて投入する）

usage: python fig03_toorpia_map.py
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
CSV = os.path.join(RES, "first3_vectors.csv")
XY = os.path.join(RES, "toorpia_first3_xy.npy")
OUT = os.path.join(ROOT, "images", "03_toorpia_map.png")

rows = list(csv.DictReader(open(CSV)))
qids = [r["qid"] for r in rows]
cats = [r["category"] for r in rows]

if os.path.exists(XY):
    xy = np.load(XY)
else:  # toorPIA API へ投入して座標を得る（要APIキー）
    from toorpia.client import toorPIA
    dim = sum(1 for k in rows[0] if k.startswith("d"))
    X = np.array([[float(r[f"d{i}"]) for i in range(dim)] for r in rows])
    rng = np.random.default_rng(42)
    Xj = X + rng.normal(0, 1, X.shape) * (X.std(0, keepdims=True) * 1e-3 + 1e-8)
    tmp = CSV.replace(".csv", "_jitter.csv")
    with open(tmp, "w") as f:
        f.write("qid,category,trial," + ",".join(f"d{i}" for i in range(dim)) + "\n")
        for k, r in enumerate(rows):
            f.write(f"{r['qid']},{r['category']},{r['trial']}," +
                    ",".join(f"{v:.5f}" for v in Xj[k]) + "\n")
    client = toorPIA()
    xy = np.asarray(client.fit_transform_csvform(
        tmp, drop_columns=["qid", "category", "trial"],
        label="LLM first-3-token hidden states (Qwen3-4B)", tag="lecture"),
        dtype=float)
    np.save(XY, xy)

catcol = {"science_true": "#2ca02c", "science_false": "#d62728",
          "contested": "#ff7f0e", "political": "#9467bd"}
catja = {"science_true": "科学・正しい命題", "science_false": "科学・誤った命題",
         "contested": "議論が分かれる", "political": "政治・センシティブ"}
qja = {"st01": "地球→太陽", "st02": "水=H2O", "st08": "太平洋最大",
       "sf01": "太陽→地球?", "sf02": "水100℃で凍る?", "sf07": "DNA三重らせん?",
       "ct01": "原発は最良?", "ct04": "AIが仕事を奪う?", "ct06": "死刑に抑止力?",
       "po01": "台湾は独立国?", "po02": "台湾は中国の一部?", "po06": "中国は民主主義?"}
off = {"sf07": (0, 14), "po01": (46, 6), "po02": (-52, 6)}

uq = list(dict.fromkeys(qids))
fig, ax = plt.subplots(figsize=(10.5, 8))
for q in uq:
    idx = [i for i, x in enumerate(qids) if x == q]
    p = xy[idx]
    c = catcol[cats[idx[0]]]
    cx, cy = p[:, 0].mean(), p[:, 1].mean()
    sp = float(np.mean(np.linalg.norm(p - p.mean(0), axis=1)))
    if sp > 0.01:  # 割れた質問: スパイダー線で所属を明示
        for x, y in p:
            ax.plot([cx, x], [cy, y], color=c, lw=0.7, alpha=0.45, zorder=1)
    ax.scatter(p[:, 0], p[:, 1], s=46, color=c, alpha=0.7,
               edgecolors="white", zorder=3)
    ax.annotate(qja[q], (cx, cy), fontsize=9.5, weight="bold",
                xytext=off.get(q, (0, 11)), textcoords="offset points",
                ha="center", zorder=5)
ax.set_title("toorPIA マップ：12問 × 各20回の「出だし数語の hidden state」（Qwen3-4B 実測）\n"
             "ほとんどの質問は20回ぶんが一点に固まる（一貫）。割れる質問だけが複数の塊に分かれる（線=同じ質問）",
             fontsize=11)
ax.set_xlabel("toorPIA-x"); ax.set_ylabel("toorPIA-y"); ax.grid(alpha=0.3)
ax.legend(handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor=v,
                          markersize=9, label=catja[k]) for k, v in catcol.items()],
          fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print(f"saved: {OUT}")
