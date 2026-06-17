#!/usr/bin/env python3
"""立場スイープ・アニメーション（ヒートマップ明滅版）:
立場 s を 反対(0)→賛成(4)→反対 とスイープすると、その立場に対応する『密度フィールドの領域』が
発光してマップ上を連続移動する。＝立場が連続軸として内部に刻まれていることを、背景ヒートマップの
明滅・推移で見せる。

各セルは平均立場(0..1)を持ち、現在の立場 s に近いセルほど明るく点灯（発光は局所密度に比例）。
出力: results/Qwen3-4B/anim_stance_field.mp4 / .gif
usage: python figures/anim_stance_field.py
"""
import json
import os
import sys

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
LNAME = ["強い反対", "反対", "中立", "賛成", "強い賛成"]
STANCE = LinearSegmentedColormap.from_list("stance", [
    (0.00, "#1f6bff"), (0.25, "#36c5ff"), (0.5, "#6e7480"),
    (0.75, "#ff9e2c"), (1.00, "#ff2740")])
plt.rcParams["font.family"] = "IPAexGothic"
plt.rcParams["text.parse_math"] = False

meta = sorted(json.load(open(os.path.join(RES, "traj_meta_stance.json"))),
              key=lambda m: (m["sent_id"], m["variant"]))
lev = np.array([LEVEL[m["sent_id"]] for m in meta])
XY = np.load(os.path.join(RES, "traj_stance_xy.npy"))
sizes = np.load(os.path.join(RES, "traj_stance_sizes.npy"))
bounds = np.cumsum([0] + list(sizes))
C = np.array([XY[bounds[i]:bounds[i + 1]].mean(0) for i in range(len(meta))])
tok_lev = np.concatenate([np.full(sizes[i], lev[i]) for i in range(len(meta))])

xmin, ymin = XY.min(0); xmax, ymax = XY.max(0)
mx, my = (xmax - xmin) * .06, (ymax - ymin) * .06
ext = [xmin - mx, xmax + mx, ymin - my, ymax + my]
rng = [[ext[0], ext[1]], [ext[2], ext[3]]]
Hc, _, _ = np.histogram2d(XY[:, 0], XY[:, 1], bins=420, range=rng)
Hl, _, _ = np.histogram2d(XY[:, 0], XY[:, 1], bins=420, range=rng, weights=tok_lev)
Cnt = gaussian_filter(Hc.T, sigma=9); Lvl = gaussian_filter(Hl.T, sigma=9)
field = np.divide(Lvl, Cnt, out=np.full_like(Lvl, 2.0), where=Cnt > 1e-6) / 4.0  # 0..1
dens = Cnt / Cnt.max()
field_rgb = STANCE(np.clip(field, 0, 1))[..., :3]   # 色は各セルの立場色（固定）

fig, ax = plt.subplots(figsize=(11, 10))
fig.patch.set_facecolor("#05060a"); ax.set_facecolor("#05060a")
# 静的: 全フィールドを薄く（常時の文脈）
base = np.dstack([field_rgb, np.clip(dens ** 0.5 * 0.16, 0, 0.2)])
ax.imshow(base, origin="lower", extent=ext, aspect="auto", interpolation="bilinear", zorder=0)
# 動的: 現在の立場 s に近い領域だけ発光
dyn_rgba = np.dstack([field_rgb, np.zeros_like(dens)])
im = ax.imshow(dyn_rgba, origin="lower", extent=ext, aspect="auto", interpolation="bilinear", zorder=1)
# 反対/賛成のアンカー（静的）
a0 = C[lev == 0].mean(0); a4 = C[lev == 4].mean(0)
ax.annotate("反対", tuple(a0), color="#37a0ff", fontsize=13, weight="bold", ha="center")
ax.annotate("賛成", tuple(a4), color="#ff5a4d", fontsize=13, weight="bold", ha="center")
ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
sm = plt.cm.ScalarMappable(cmap=STANCE, norm=plt.Normalize(0, 4)); sm.set_array([])
cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.015, ticks=range(5))
cb.ax.set_yticklabels(LNAME); cb.set_label("立場", color="white")
cb.ax.yaxis.set_tick_params(color="white"); plt.setp(cb.ax.get_yticklabels(), color="white")
ax.set_title("立場をスイープ：内部状態の地形が反対↔賛成へ明滅・推移する", fontsize=18,
             weight="bold", color="white", pad=22)
sub = fig.text(0.5, 0.045, "", ha="center", fontsize=13, color="#cfd6e2")

N = 65
sweep = np.concatenate([np.linspace(0, 4, N), np.linspace(4, 0, N)])
WID = 0.11
cbline = [None]


def update(f):
    s = sweep[f]; sn = s / 4.0
    hl = np.exp(-((field - sn) / WID) ** 2)
    a = np.clip(dens ** 0.42 * hl * 1.15, 0, 0.98)
    rgba = np.dstack([field_rgb, a])
    im.set_data(rgba)
    if cbline[0] is not None:
        cbline[0].remove()
    cbline[0] = cb.ax.axhline(s, color="white", lw=2.6, alpha=0.95)
    sub.set_text(f"立場 s = {s:.1f}（{LNAME[int(round(s))]}寄り）　— 点灯領域が反対↔賛成へ連続移動")
    return [im]


anim = animation.FuncAnimation(fig, update, frames=len(sweep), interval=42, blit=False)
mp4 = os.path.join(RES, "anim_stance_field.mp4")
anim.save(mp4, writer=animation.FFMpegWriter(fps=24, bitrate=3600), dpi=120)
print("saved:", mp4)
