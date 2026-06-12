#!/usr/bin/env python3
"""教材図 images/07_tree_limit.png を生成する（スライド14・木展開の限界）。

左: 木展開の枝数 vs 深さ（3課題）。オープンエンド課題では冒頭が質問の
    復唱でほぼ決定的（枝1本）、分岐は7〜13語目から始まり、以後は毎語
    ×1.2〜1.3 で増殖する＝回答全文の深さでは木は計算不能になる。
右: 複雑課題への14回のフル回答の hidden state（全文平均）をtoorPIAで2D化。
    文字列としては多数の枝に分岐していても、意味では「時系列分析」の
    1クラスタ＋例外1（顧客分析）に束ねられる。
    ※意味ラベルは回答全文を読んで付与した目視判定。

入力: results/Qwen3-4B/complex_task.json, complex_task_vectors.npz
      （12_complex_task.py の出力）, toorpia_sales_xy.npy（toorPIA座標）

usage: python fig07_tree_limit.py
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
OUT = os.path.join(ROOT, "images", "07_tree_limit.png")

d = json.load(open(os.path.join(RES, "complex_task.json")))
vec = np.load(os.path.join(RES, "complex_task_vectors.npz"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))

# ---- 左: 枝数 vs 深さ ----
style = {"simple_st01": ("#2ca02c", "単純質問（地球→太陽）"),
         "complex_sales": ("#d62728", "複雑課題（売上分析の提案）"),
         "complex_cause": ("#ff7f0e", "複雑課題（売上急落の原因）")}
for name, (c, lb) in style.items():
    cur = d["tree_curves"][name]["curve"]
    ax1.plot([r["depth"] for r in cur], [r["n_branches"] for r in cur],
             marker="o", ms=4, color=c, label=lb, lw=2)
# 外挿: sales の実測成長率(末尾5深さ)で回答全文の深さへ
cur = d["tree_curves"]["complex_sales"]["curve"]
bs = [r["n_branches"] for r in cur]
growth = (bs[-1] / bs[-6]) ** (1 / 5)
xs = np.arange(cur[-1]["depth"], 41)
ax1.plot(xs, bs[-1] * growth ** (xs - xs[0]), ls="--", color="#d62728",
         alpha=0.6, lw=1.5)
ax1.annotate(f"実測成長率 ×{growth:.2f}/語 で外挿\n→ 回答全文(90語)では桁外れ\n（サンプリングは線形のまま）",
             xy=(40, bs[-1] * growth ** (40 - xs[0])), xytext=(23, 300),
             fontsize=9.5, color="#a33",
             arrowprops=dict(arrowstyle="->", color="#a33"),
             bbox=dict(boxstyle="round", fc="#ffecec", ec="#c88"))
ax1.axhspan(0.8, 1.2, color="#bbb", alpha=0.25)
ax1.text(4, 1.35, "復唱（エコー）区間：枝1本のまま\n＝出だしの数語に際どさは現れない",
         fontsize=9, color="#555")
ax1.set_yscale("log")
ax1.set_xlabel("木の深さ（出力トークン数）")
ax1.set_ylabel("枝の本数（対数軸）")
ax1.set_title("木展開のコスト：分岐は深部で始まり、以後 指数的に増殖", fontsize=11)
ax1.legend(fontsize=9, loc="upper left")
ax1.grid(alpha=0.3)

# ---- 右: 複雑課題の意味クラスタ ----
V = vec["complex_sales"]
P = np.load(os.path.join(RES, "toorpia_sales_xy.npy"))  # toorPIA座標
# 意味ラベル（回答全文を読んで付与した目視判定）
is_ts = [("time se" in t.lower() or "time-se" in t.lower() or "trend" in t.lower()
          or "sales tr" in t.lower())
         for t in d["samples"]["complex_sales"]]
ts_idx = [i for i, b in enumerate(is_ts) if b]
ts = ts_idx
ot = [i for i, b in enumerate(is_ts) if not b]
rng = np.random.default_rng(1)
ax2.scatter(P[ts, 0] + rng.normal(0, P[:, 0].std() * 0.02, len(ts)),
            P[ts, 1] + rng.normal(0, P[:, 1].std() * 0.02, len(ts)),
            s=130, color="#d62728", alpha=0.65, edgecolors="white",
            label=f"「時系列/トレンド分析をせよ」×{len(ts)}")
ax2.scatter(P[ot, 0], P[ot, 1], s=160, color="#1f77b4", marker="^",
            alpha=0.9, edgecolors="white",
            label=f"「顧客セグメント分析をせよ」×{len(ot)}")
ax2.set_title("同じ複雑課題への14回のフル回答（全文の hidden state）\n"
              "文字列は多数に分岐しても、意味のまとまりに束ねられる",
              fontsize=10.5)
ax2.set_xlabel("toorPIAによる2次元化（使用モデル: Qwen3-4B）")

# 赤の2つの小塊の正体（時系列「分析」と時系列「分解」）を注記
deco = [i for i in ts_idx
        if "decomposition" in d["samples"]["complex_sales"][i].lower()]
ana = [i for i in ts_idx if i not in deco]
ax2.annotate(f"「時系列の“分析”をせよ」×{len(ana)}",
             xy=(P[ana, 0].mean(), P[ana, 1].mean()),
             xytext=(35, -35), textcoords="offset points",
             fontsize=9, color="#a33",
             arrowprops=dict(arrowstyle="->", color="#a33"))
ax2.annotate(f"「時系列の“分解”をせよ」×{len(deco)}\n（同じ時系列系。提案の具体性の違い）",
             xy=(P[deco, 0].mean(), P[deco, 1].mean()),
             xytext=(-30, 40), textcoords="offset points",
             fontsize=9, color="#a33",
             arrowprops=dict(arrowstyle="->", color="#a33"))
ax2.legend(fontsize=9.5, loc="upper center")
ax2.grid(alpha=0.3)
ax2.text(0.5, 0.04, "※意味ラベルは回答全文を読んで確認（目視判定）",
         transform=ax2.transAxes, ha="center", fontsize=8.5, color="#777")

fig.suptitle("複雑な課題では：木は深部の分岐で破綻し、意味の束ねは hidden 空間が担う（Qwen3-4B 実測）",
             fontsize=12, weight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT, dpi=140)
print(f"saved: {OUT}")
