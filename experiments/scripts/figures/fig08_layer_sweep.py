#!/usr/bin/env python3
"""教材図 images/08_layer_sweep.png を生成する（スライド17・開かれた問い）。

「意味・立場はどの層で最も純粋に表現されるか」の予備実験（13_layer_sweep.py）。
左: 立場・意味の分離度（高いほど良い）の層プロファイル
右: 言い回しノイズ（同一立場内の散らばり÷立場間距離。低いほど
    言い回しの違いが畳み込まれている）の層プロファイル
→ 指標ごとに最適層が異なり、単一の答えはまだ無い＝開かれた研究課題。

入力: results/Qwen3-4B/layer_sweep.json

usage: python fig08_layer_sweep.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "IPAexGothic"
EXP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(EXP)
OUT = os.path.join(ROOT, "images", "08_layer_sweep.png")

d = json.load(open(os.path.join(EXP, "results", "Qwen3-4B", "layer_sweep.json")))
ms = d["metrics"][1:]  # 0=埋め込み層は除外
xs = [m["layer"] for m in ms]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

ax1.plot(xs, [m["sf07_contrast"] for m in ms], marker="o", ms=4,
         color="#d62728", lw=2, label="立場の分離（DNA: Yes派 vs 二重らせん派）")
ax1.plot(xs, [m["sales_contrast"] for m in ms], marker="s", ms=4,
         color="#ff7f0e", lw=2, label="意味の分離（売上課題: 時系列 vs 顧客分析）")
ax1.axvspan(27, 30, color="#ff7f0e", alpha=0.12)
ax1.text(28.5, 1.1, "意味の分離は\nL27-30がピーク", ha="center", fontsize=9,
         color="#b60")
ax1.set_xlabel("層（最終層=36）")
ax1.set_ylabel("分離度 = 立場間の距離 ÷ 同一立場内のブレ（高いほど良い）")
ax1.set_title("立場・意味の分離はどの層が得意か", fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

ax2.plot(xs, [m["po06_rel_noise"] for m in ms], marker="o", ms=4,
         color="#9467bd", lw=2,
         label="言い回しノイズ（中国民主主義: 同一立場の散らばり、相対値）")
ax2.axvspan(15, 24, color="#9467bd", alpha=0.12)
ax2.text(19.5, max(m["po06_rel_noise"] for m in ms) * 0.97,
         "中間層(L15-24)が\n言い回しを最も畳み込む\n（最終層より約3割低い）",
         ha="center", fontsize=9, color="#656")
ax2.set_xlabel("層（最終層=36）")
ax2.set_ylabel("言い回しノイズ（低いほど良い）")
ax2.set_title("言い回しの違いを「同じ意味」として畳み込めるのはどの層か", fontsize=11)
ax2.legend(fontsize=8.5)
ax2.grid(alpha=0.3)

fig.suptitle("予備実験：指標ごとに最適な層が違う ― 「どの層で意味を測るべきか」はまだ開かれた問い（Qwen3-4B）",
             fontsize=12, weight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT, dpi=140)
print(f"saved: {OUT}")
