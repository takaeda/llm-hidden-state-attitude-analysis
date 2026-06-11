#!/usr/bin/env python3
"""教材図 images/04_consistency_standard.png を生成する（スライド12）。

4モデルの一貫性評価の3パネル図:
 ① d′（物差しの分解能）と信用閾値
 ② 規格化非一貫性 vs 事実正答率（一貫性と正しさは別軸）
 ③ 質問カテゴリ別の規格化非一貫性ヒートマップ（配備マップ。物差し有効モデルのみ）

入力: results/<model>/consistency.json（08_consistency_metric.py の出力）
      results/<model>/profile.json（07_model_profile.py の出力。正答率）

usage: python fig04_consistency_standard.py
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
OUT = os.path.join(ROOT, "images", "04_consistency_standard.png")

models = ["Qwen3-4B", "Phi-4-mini-instruct", "Mistral-7B-Instruct-v0.3",
          "DeepSeek-R1-Distill-Qwen-1.5B"]
short = {"Qwen3-4B": "Qwen3-4B", "Phi-4-mini-instruct": "Phi-4-mini",
         "Mistral-7B-Instruct-v0.3": "Mistral-7B",
         "DeepSeek-R1-Distill-Qwen-1.5B": "DeepSeek-1.5B"}
C = {m: json.load(open(os.path.join(EXP, "results", m, "consistency.json")))
     for m in models}
A = {m: json.load(open(os.path.join(EXP, "results", m, "profile.json")))
     ["factual_accuracy"] for m in models}

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 5.2),
                                    gridspec_kw={"width_ratios": [1, 1.1, 1.2]})

# ① d′ 安全弁
dp = [C[m]["d_prime_validity"] for m in models]
cols = ["#2ca02c" if C[m]["valid"] else "#cccccc" for m in models]
ax1.bar(range(4), dp, color=cols, alpha=0.9)
ax1.axhline(3.0, color="#c00", ls="--", lw=1.5)
ax1.text(3.4, 3.15, "信用閾値 d'=3", color="#c00", fontsize=9, ha="right")
ax1.set_xticks(range(4))
ax1.set_xticklabels([short[m] for m in models], rotation=20, ha="right", fontsize=9)
ax1.set_ylabel("d′（物差しの分解能）")
ax1.set_title("① 安全弁：そもそも一貫性を測れる土台があるか\n緑=物差し有効 / 灰=疑わしい", fontsize=10)
for i, m in enumerate(models):
    ax1.text(i, dp[i] + 0.1, f"{dp[i]:.1f}", ha="center", fontsize=9)
ax1.grid(axis="y", alpha=0.3)

# ② 一貫性 vs 正答率
for m in models:
    inc = C[m]["norm_inconsistency_overall"]
    acc = A[m]
    valid = C[m]["valid"]
    ax2.scatter(inc, acc, s=200, color="#2ca02c" if valid else "#bbb",
                edgecolors="black" if valid else "#999", lw=1.5, zorder=3,
                marker="o" if valid else "X")
    ax2.annotate(short[m], (inc, acc), fontsize=9.5,
                 xytext=(8, 4), textcoords="offset points")
ax2.set_xlabel("← 一貫している      規格化した非一貫性      バラつく →")
ax2.set_ylabel("事実正答率（参考・別軸）")
ax2.set_title("② 一貫性と正しさは別の軸\n(×=物差し疑わしく評価対象外)", fontsize=10)
ax2.axhline(0.5, color="#ccc", ls=":", lw=1)
ax2.grid(alpha=0.3)
ax2.annotate("Mistralは高精度だが\nQwenより一貫性は低い\n→定型業務ならQwen",
             xy=(C["Mistral-7B-Instruct-v0.3"]["norm_inconsistency_overall"],
                 A["Mistral-7B-Instruct-v0.3"]),
             xytext=(0.9, 0.72), fontsize=8.5,
             arrowprops=dict(arrowstyle="->", color="#555"),
             bbox=dict(boxstyle="round", fc="#eef7ff", ec="#88a"))

# ③ 配備マップ（物差し有効モデルのみ）
valid_models = [m for m in models if C[m]["valid"]]
cats = ["science_true", "science_false", "contested", "political"]
catja = ["科学(正)", "科学(偽)", "議論", "政治"]
M = np.array([[C[m]["norm_inconsistency_by_category"][c] for c in cats]
              for m in valid_models])
im = ax3.imshow(M, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1.0)
ax3.set_xticks(range(4)); ax3.set_xticklabels(catja, fontsize=9)
ax3.set_yticks(range(len(valid_models)))
ax3.set_yticklabels([short[m] for m in valid_models])
for i in range(len(valid_models)):
    for j in range(4):
        ax3.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=10)
ax3.set_title("③ 配備マップ：どの領域なら一貫して使えるか\n(緑=一貫/赤=バラつく。物差し有効モデルのみ)", fontsize=10)
plt.colorbar(im, ax=ax3, fraction=0.04, label="規格化した非一貫性")
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print(f"saved: {OUT}")
