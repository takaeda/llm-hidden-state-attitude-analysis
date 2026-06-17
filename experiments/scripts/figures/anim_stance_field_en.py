#!/usr/bin/env python3
"""立場スイープ・アニメーション（英語字幕・高dpi・代表意見つき）。
背景の密度フィールドが、現在の立場 s に対応する領域だけ発光して反対↔賛成へ推移。
画面上部に、その立場の代表的な回答（英文）を印象的に表示。

出力: results/Qwen3-4B/anim_stance_en.mp4 / .gif
usage: python figures/anim_stance_field_en.py
"""
import json
import os
import sys
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "results", "Qwen3-4B")
LEVEL = {"s0_strong_anti": 0, "s1_anti": 1, "s2_neutral": 2, "s3_pro": 3, "s4_strong_pro": 4}
TAGS = ["STRONGLY OPPOSE", "OPPOSE", "NEUTRAL", "SUPPORT", "STRONGLY SUPPORT"]
QUOTES = [
    "“No — building nuclear power plants is not the right energy policy; it poses significant risks.”",
    "“A contentious choice: heavy financial, environmental and safety risks that often outweigh the benefits.”",
    "“It can be a viable part of energy policy — but it also brings significant costs and long-term concerns.”",
    "“A viable policy for cutting carbon and meeting demand — with careful planning, safety and waste management.”",
    "“Yes — a right and necessary policy: reliable, low-carbon power that helps combat climate change.”",
]
STANCE = LinearSegmentedColormap.from_list("stance", [
    (0.00, "#1f6bff"), (0.25, "#36c5ff"), (0.5, "#8a909c"),
    (0.75, "#ff9e2c"), (1.00, "#ff2740")])
plt.rcParams["font.family"] = "IPAexGothic"
plt.rcParams["text.parse_math"] = False


def bright(rgb):  # 暗い中立色を持ち上げて読みやすく
    r, g, b, _ = rgb
    return (min(1, r * 0.5 + 0.5), min(1, g * 0.5 + 0.5), min(1, b * 0.5 + 0.5))


meta = sorted(json.load(open(os.path.join(RES, "traj_meta_stance.json"))),
              key=lambda m: (m["sent_id"], m["variant"]))
lev = np.array([LEVEL[m["sent_id"]] for m in meta])
XY = np.load(os.path.join(RES, "traj_stance_xy.npy"))
sizes = np.load(os.path.join(RES, "traj_stance_sizes.npy"))
bounds = np.cumsum([0] + list(sizes))
C = np.array([XY[bounds[i]:bounds[i + 1]].mean(0) for i in range(len(meta))])
tok_lev = np.concatenate([np.full(sizes[i], lev[i]) for i in range(len(meta))])
A = np.c_[C, np.ones(len(C))]; coef, *_ = np.linalg.lstsq(A, lev, rcond=None)
w = coef[:2] / (np.linalg.norm(coef[:2]) + 1e-9); rho_t = C @ w
ra = np.argsort(np.argsort(lev)); rb = np.argsort(np.argsort(rho_t))
rho = float(np.corrcoef(ra, rb)[0, 1])

xmin, ymin = XY.min(0); xmax, ymax = XY.max(0)
mx, my = (xmax - xmin) * .06, (ymax - ymin) * .06
ext = [xmin - mx, xmax + mx, ymin - my, ymax + my]
rng = [[ext[0], ext[1]], [ext[2], ext[3]]]
Hc, _, _ = np.histogram2d(XY[:, 0], XY[:, 1], bins=420, range=rng)
Hl, _, _ = np.histogram2d(XY[:, 0], XY[:, 1], bins=420, range=rng, weights=tok_lev)
Cnt = gaussian_filter(Hc.T, sigma=9); Lvl = gaussian_filter(Hl.T, sigma=9)
field = np.divide(Lvl, Cnt, out=np.full_like(Lvl, 2.0), where=Cnt > 1e-6) / 4.0
dens = Cnt / Cnt.max()
field_rgb = STANCE(np.clip(field, 0, 1))[..., :3]

fig, ax = plt.subplots(figsize=(12.5, 12.6))
fig.patch.set_facecolor("#05060a"); ax.set_facecolor("#05060a")
fig.subplots_adjust(left=0.02, right=0.83, top=0.74, bottom=0.06)  # 上部=字幕 / 右=カラーバー余白
base = np.dstack([field_rgb, np.clip(dens ** 0.5 * 0.16, 0, 0.2)])
ax.imshow(base, origin="lower", extent=ext, aspect="auto", interpolation="bilinear", zorder=0)
im = ax.imshow(np.dstack([field_rgb, np.zeros_like(dens)]), origin="lower", extent=ext,
               aspect="auto", interpolation="bilinear", zorder=1)
a0 = C[lev == 0].mean(0); a4 = C[lev == 4].mean(0)
ax.annotate("Oppose", tuple(a0), color="#37a0ff", fontsize=13, weight="bold", ha="center")
ax.annotate("Support", tuple(a4), color="#ff5a4d", fontsize=13, weight="bold", ha="center")
ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
sm = plt.cm.ScalarMappable(cmap=STANCE, norm=plt.Normalize(0, 4)); sm.set_array([])
cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.015, ticks=range(5))
cb.ax.set_yticklabels(TAGS, fontsize=9); cb.set_label("Stance", color="white")
cb.ax.yaxis.set_tick_params(color="white"); plt.setp(cb.ax.get_yticklabels(), color="white")

# 上部テキスト: コンテキスト見出し + タグ + 代表意見(引用)
fig.text(0.5, 0.95, "Stance is written into the internal state as a continuous gradient",
         ha="center", fontsize=15, weight="bold", color="white")
fig.text(0.5, 0.915, "Sweeping oppose↔support over “Is building nuclear power the right energy policy?”  ·  "
         f"Qwen3-4B · internal states mapped by toorPIA · continuity ρ = {rho:.2f}",
         ha="center", fontsize=11, color="#9aa3b2")
tag_t = fig.text(0.5, 0.86, "", ha="center", fontsize=16, weight="bold", color="white")
quote_t = fig.text(0.5, 0.80, "", ha="center", va="top", fontsize=17, style="italic", color="white")
sub = fig.text(0.5, 0.025, "", ha="center", fontsize=12.5, color="#cfd6e2")

N = 70
sweep = np.concatenate([np.linspace(0, 4, N), np.linspace(4, 0, N)])
WID = 0.11
cbline = [None]


def update(f):
    s = sweep[f]; sn = s / 4.0; k = int(round(s))
    hl = np.exp(-((field - sn) / WID) ** 2)
    a = np.clip(dens ** 0.42 * hl * 1.15, 0, 0.98)
    im.set_data(np.dstack([field_rgb, a]))
    if cbline[0] is not None:
        cbline[0].remove()
    cbline[0] = cb.ax.axhline(s, color="white", lw=2.6, alpha=0.95)
    cc = STANCE(sn)
    tag_t.set_text(f"●  {TAGS[k]}   (s = {s:.1f})"); tag_t.set_color(cc)
    quote_t.set_text("\n".join(textwrap.wrap(QUOTES[k], width=64)))
    quote_t.set_color(bright(cc))
    sub.set_text(f"The lit region of the internal-state landscape moves continuously from Oppose to Support")
    return [im]


anim = animation.FuncAnimation(fig, update, frames=len(sweep), interval=42, blit=False)
mp4 = os.path.join(RES, "anim_stance_en.mp4")
anim.save(mp4, writer=animation.FFMpegWriter(fps=24, bitrate=6000), dpi=200)
print(f"rho={rho:.3f}  saved: {mp4}")
