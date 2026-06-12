#!/usr/bin/env python3
"""本編図 images/e01_chain.png を生成する（平易版・鎖の図）。

「AIは一語ずつサイコロを振って文を作る。だから答えはゆらぐ」だけを伝える。
データ入力なし（模式図）。

usage: python fig_e01_chain.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

plt.rcParams["font.family"] = "IPAexGothic"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "images", "e01_chain.png")

fig, ax = plt.subplots(figsize=(13, 5.2))
ax.set_xlim(0, 13.5); ax.set_ylim(1.6, 6.4); ax.axis("off")
ax.text(6.75, 6.15, "AIは一語ずつ、サイコロを振りながら文を作っている",
        ha="center", fontsize=16, weight="bold")


def node(x, y, c, r=0.24, tx="", fs=9):
    ax.add_patch(Circle((x, y), r, fc=c, ec="white", lw=1.5, zorder=3))
    if tx:
        ax.text(x, y, tx, ha="center", va="center", fontsize=fs,
                color="white", weight="bold", zorder=4)


def edge(x1, y1, x2, y2, c="#999", lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, lw=lw, color=c, zorder=2))


yC = 4.4
px = [0.7, 1.6, 2.5, 3.4, 4.3]
labs = ["DNA", "は", "三重", "らせん", "？"]
for i, (x, l) in enumerate(zip(px, labs)):
    node(x, yC, "#9aa", tx=l, fs=8)
    if i > 0:
        edge(px[i - 1] + 0.24, yC, x - 0.24, yC, "#bbb")
ax.add_patch(plt.Rectangle((0.4, 3.85), 4.35, 1.1, fill=False, ec="#888",
                           lw=1.6, ls="--"))
ax.text(2.55, 3.65, "質問（こちらが入力するので、毎回同じ）",
        ha="center", va="top", fontsize=11, color="#666")

fx = 5.7
edge(4.3 + 0.24, yC, fx - 0.32, yC, "#c60", 2)
node(fx, yC, "#e67", r=0.32)
ax.text(fx, 5.35, "答えの最初の一語を選ぶ\n＝最初のサイコロ", ha="center",
        fontsize=11.5, color="#c33", weight="bold")
ax.annotate("ゆらぎはここから生まれる", xy=(fx, yC - 0.34),
            xytext=(fx + 0.3, 2.7), fontsize=12, color="#c33", weight="bold",
            ha="center", arrowprops=dict(arrowstyle="->", color="#c33"))

l1 = [(7.1, 5.2, "Yes", "#3a8"), (7.1, 3.6, "DNA", "#e90")]
for x, y, t, c in l1:
    node(x, y, c, tx=t, fs=9)
    edge(fx + 0.32, yC, x - 0.24, y, c, 1.8)
l2 = [(8.5, 5.6, "#3a8", l1[0]), (8.5, 4.8, "#3a8", l1[0]),
      (8.5, 4.0, "#e90", l1[1]), (8.5, 3.2, "#e90", l1[1])]
for x, y, c, par in l2:
    node(x, y, c, r=0.19)
    edge(par[0] + 0.24, par[1], x - 0.19, y, c, 1.3)
for (x, y, c, _) in l2:
    for dy in (0.36, -0.36):
        node(x + 1.3, y + dy, c, r=0.13)
        edge(x + 0.19, y, x + 1.3 - 0.13, y + dy, c, 1.0)
        ax.text(x + 1.62, y + dy, "…", fontsize=11, color=c, va="center")

ax.text(11.9, 4.4, "サイコロの出目しだいで\n文章全体が変わる\n＝答えがゆらぐ",
        ha="center", fontsize=12, color="#c60", weight="bold")
ax.text(6.75, 2.05, "選んだ一語が次の入力になるので、分かれ道がどんどん積み重なっていく",
        ha="center", fontsize=12.5, weight="bold",
        bbox=dict(boxstyle="round,pad=0.45", fc="#fffbe0", ec="#aa8"))
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"saved: {OUT}")
