#!/usr/bin/env python3
"""本編図 images/e04_stability.png を生成する（平易版・安定性マップ）。

「AIごとに、安定して答えられる話題が違う ＝ 文章を読まなくても
モデル選びの地図が作れる」だけを伝える。数値は出さず色と記号で。

入力: results/<model>/consistency.json（Qwen3-4B, Mistral-7B）

usage: python fig_e04_stability.py
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
OUT = os.path.join(ROOT, "images", "e04_stability.png")

models = ["Qwen3-4B", "Mistral-7B-Instruct-v0.3"]
short = {"Qwen3-4B": "AIモデル A\n(Qwen3-4B)",
         "Mistral-7B-Instruct-v0.3": "AIモデル B\n(Mistral-7B)"}
cats = ["science_true", "science_false", "contested", "political"]
catja = ["教科書どおり\nの質問", "ひっかけ\n質問", "意見が割れる\n話題", "政治の\n話題"]

M = np.array([[json.load(open(os.path.join(EXP, "results", m,
              "consistency.json")))["norm_inconsistency_by_category"][c]
              for c in cats] for m in models])


def grade(v):
    if v < 0.25:
        return "◎ 安定", "#bfe6bf"
    if v < 0.7:
        return "○ まずまず", "#fdf2c0"
    return "△ ゆれる", "#f6c1b0"


fig, ax = plt.subplots(figsize=(10.5, 4.6))
for i in range(len(models)):
    for j in range(len(cats)):
        g, c = grade(M[i, j])
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fc=c,
                                   ec="white", lw=3))
        ax.text(j, i, g, ha="center", va="center", fontsize=15, weight="bold",
                color="#333")
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(1.5, -0.5)
ax.set_xticks(range(4)); ax.set_xticklabels(catja, fontsize=12)
ax.set_yticks(range(2)); ax.set_yticklabels([short[m] for m in models],
                                            fontsize=12)
ax.set_title("AIごとの「安定して答えられる話題」マップ\n"
             "― 答えの文章を1行も読まずに、頭の中のばらつきだけで作った ―",
             fontsize=14, weight="bold")
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print(f"saved: {OUT}")
