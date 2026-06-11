#!/usr/bin/env python3
"""教材図 images/06_validation.png を生成する（スライド13）。

検証(a): 出だし数語の hidden state の広がり（読まない）と、
別系統の文埋め込みで測った回答文の意味的な広がり（読む）の散布図。
Qwen3-4B / Mistral-7B の2パネル、Spearman相関を表題に示す。

入力: results/validation_a.json（10_validate.py の出力）

usage: python fig06_validation.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "IPAexGothic"
EXP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(EXP)
OUT = os.path.join(ROOT, "images", "06_validation.png")

v = json.load(open(os.path.join(EXP, "results", "validation_a.json")))
catcol = {"science_true": "#2ca02c", "science_false": "#d62728",
          "contested": "#ff7f0e", "political": "#9467bd"}
catja = {"science_true": "科学・正しい命題", "science_false": "科学・誤った命題",
         "contested": "議論が分かれる", "political": "政治・センシティブ"}
models = [("Qwen3-4B", "Qwen3-4B"), ("Mistral-7B-Instruct-v0.3", "Mistral-7B")]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
for ax, (tag, short) in zip(axes, models):
    d = v[tag]
    for r in d["rows"]:
        ax.scatter(r["hidden_spread"], r["semantic_spread"], s=70,
                   color=catcol[r["category"]], alpha=0.75,
                   edgecolors="white", zorder=3)
    ax.set_xlabel("【読まない】出だし数語のhidden stateの広がり")
    ax.set_ylabel("【読む】回答文の意味的な広がり\n(別系統の埋め込みモデル)")
    ax.set_title(f"{short}: ρ={d['spearman']} (p={d['p_value']:.1g})", fontsize=11)
    ax.grid(alpha=0.3)
axes[0].legend(handles=[Line2D([0], [0], marker='o', color='w',
               markerfacecolor=catcol[k], markersize=9, label=catja[k])
               for k in catcol], fontsize=8.5, loc="lower right")
fig.suptitle("検証(a): hidden stateの広がり（読まない）は、意味的な一貫性（読む）を代理できるか\n"
             "→ 有意な正の相関。出力を読まずに一貫性を測れることを支持（出だし限定ゆえ中程度）",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(OUT, dpi=140)
print(f"saved: {OUT}")
