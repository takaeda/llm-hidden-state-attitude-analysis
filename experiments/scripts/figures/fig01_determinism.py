#!/usr/bin/env python3
"""教材図 images/01_determinism.png を生成する（スライド4）。

「生成はトークンの鎖」の概念図。入力区間は固定（ゆらがない）、
最初の出力トークンの選択でゆらぎが入り、選択が次の入力になって
枝分かれが積み重なる、という構造を描く。データ入力は不要（純粋な模式図）。

usage: python fig01_determinism.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

plt.rcParams["font.family"] = "IPAexGothic"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "images", "01_determinism.png")

fig, ax = plt.subplots(figsize=(13.5, 5.8))
ax.set_xlim(0, 13.5); ax.set_ylim(1.4, 6.7); ax.axis("off")
ax.text(5.6, 6.5, "生成は「トークンの鎖」 ― どのノードにも内部状態がある",
        ha="center", fontsize=13.5, weight="bold")


def node(x, y, c, r=0.22, tx="", fs=8, tc="white"):
    ax.add_patch(Circle((x, y), r, fc=c, ec="white", lw=1.5, zorder=3))
    if tx:
        ax.text(x, y, tx, ha="center", va="center", fontsize=fs,
                color=tc, weight="bold", zorder=4)


def edge(x1, y1, x2, y2, c="#999", lw=1.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=10, lw=lw, color=c, zorder=2))


yC = 4.6
px = [0.6, 1.4, 2.2, 3.0, 3.8]
labs = ["D", "NA", "は", "三重", "?"]
for i, (x, l) in enumerate(zip(px, labs)):
    node(x, yC, "#9aa", tx=l, fs=8)
    if i > 0:
        edge(px[i - 1] + 0.22, yC, x - 0.22, yC, "#bbb")
ax.add_patch(plt.Rectangle((0.35, 4.1), 3.7, 1.1, fill=False, ec="#888",
                           lw=1.6, ls="--"))
ax.text(2.2, 3.95, "入力区間：トークンは外から与えられ毎回同じ\n→ ここの内部状態は固定（ゆらがない）",
        ha="center", va="top", fontsize=9.5, color="#666")

fx = 5.1
edge(3.8 + 0.22, yC, fx - 0.30, yC, "#c60", 2)
node(fx, yC, "#e67", r=0.30)
ax.text(fx, 5.5, "最初の出力トークン\n＝最初の『選択』", ha="center",
        fontsize=9.5, color="#c33", weight="bold")
ax.annotate("ゆらぎの源はこの『選ぶ』一点だけ", xy=(fx, yC - 0.30),
            xytext=(fx + 0.2, 3.5), fontsize=10, color="#c33", weight="bold",
            ha="center", arrowprops=dict(arrowstyle="->", color="#c33"))

l1 = [(6.5, 5.25, "Yes", "#3a8"), (6.5, 3.95, "DNA", "#e90")]
for x, y, t, c in l1:
    node(x, y, c, tx=t, fs=8)
    edge(fx + 0.30, yC, x - 0.22, y, c, 1.6)
l2 = [(7.8, 5.7, "#3a8", l1[0]), (7.8, 4.9, "#3a8", l1[0]),
      (7.8, 4.35, "#e90", l1[1]), (7.8, 3.55, "#e90", l1[1])]
for x, y, c, par in l2:
    node(x, y, c, r=0.18)
    edge(par[0] + 0.22, par[1], x - 0.18, y, c, 1.2)
for (x, y, c, _) in l2:
    for dy in (0.38, -0.38):
        node(x + 1.2, y + dy, c, r=0.12)
        edge(x + 0.18, y, x + 1.2 - 0.12, y + dy, c, 1.0)
        ax.text(x + 1.5, y + dy, "…", fontsize=10, color=c, va="center")

ax.text(11.7, 4.6, "→ 文全体が\nゆらぐ", ha="center", fontsize=11,
        color="#c60", weight="bold")
ax.text(8.0, 2.8, "出力区間：これらのノードにも内部状態がある。前に選ばれた語が毎回違うので、ここの内部状態も毎回変わる（＝ゆらぐ）",
        ha="center", fontsize=10, color="#2a6", weight="bold")
ax.text(6.75, 2.0, "内部状態は乱数を持たない（入力が決まれば一意）。ゆらぎは『選ぶ』一点で生じ、その語が次の入力になって、出力区間の内部状態が次々と変わっていく。",
        ha="center", fontsize=10.5, weight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fffbe0", ec="#aa8"))
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"saved: {OUT}")
