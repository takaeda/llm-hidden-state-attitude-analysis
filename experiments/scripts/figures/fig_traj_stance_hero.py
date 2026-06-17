#!/usr/bin/env python3
"""予備実験(立場5段)の結果を、1枚の“ヒーロー図”として魅せる版。
黒背景に、反対=青〜賛成=赤の滑らかな『立場の地形(stance-field)』を光らせ、
各回答の重心を同じ連続カラーで散らし、5段の重心を輝点として並べる。

stance-field = 各セル近傍の点の平均立場レベル(0..4)を Nadaraya-Watson(=密度加重平均)で算出し、
発光は局所密度に比例（空白は黒）。立場が空間的に連続して変化することを直接見せる。

usage: python figures/fig_traj_stance_hero.py   （座標は traj_stance_xy.npy を再利用）
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "results", "Qwen3-4B")
LEVEL = {"s0_strong_anti": 0, "s1_anti": 1, "s2_neutral": 2, "s3_pro": 3, "s4_strong_pro": 4}
LNAME = ["強い反対", "反対", "中立", "賛成", "強い賛成"]
# 反対=エレクトリックブルー → 中立=ミュート → 賛成=ホットレッド（両極が発光）
STANCE = LinearSegmentedColormap.from_list("stance", [
    (0.00, "#1f6bff"), (0.25, "#36c5ff"), (0.5, "#6e7480"),
    (0.75, "#ff9e2c"), (1.00, "#ff2740")])


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    plt.rcParams["font.family"] = "IPAexGothic"
    plt.rcParams["text.parse_math"] = False
    meta = json.load(open(os.path.join(RES, "traj_meta_stance.json")))
    meta = sorted(meta, key=lambda m: (m["sent_id"], m["variant"]))   # fit時と同順
    lev = np.array([LEVEL[m["sent_id"]] for m in meta])
    XY = np.load(os.path.join(RES, "traj_stance_xy.npy"))
    sizes = np.load(os.path.join(RES, "traj_stance_sizes.npy"))
    bounds = np.cumsum([0] + list(sizes))
    assert bounds[-1] == len(XY)
    curves = [XY[bounds[i]:bounds[i + 1]] for i in range(len(meta))]
    C = np.array([c.mean(0) for c in curves])               # 文書重心
    tok_lev = np.concatenate([np.full(sizes[i], lev[i]) for i in range(len(meta))])

    # 立場軸（連続性指標）
    A = np.c_[C, np.ones(len(C))]; coef, *_ = np.linalg.lstsq(A, lev, rcond=None)
    w = coef[:2] / (np.linalg.norm(coef[:2]) + 1e-9)
    rho = spearman(lev, C @ w)

    # --- stance-field（密度加重の平均立場） ---
    xmin, ymin = XY.min(0); xmax, ymax = XY.max(0)
    mx, my = (xmax - xmin) * .06, (ymax - ymin) * .06
    ext = [xmin - mx, xmax + mx, ymin - my, ymax + my]
    B = 420
    rng = [[ext[0], ext[1]], [ext[2], ext[3]]]
    Hc, _, _ = np.histogram2d(XY[:, 0], XY[:, 1], bins=B, range=rng)
    Hl, _, _ = np.histogram2d(XY[:, 0], XY[:, 1], bins=B, range=rng, weights=tok_lev)
    s = 9
    Cnt = gaussian_filter(Hc.T, sigma=s); Lvl = gaussian_filter(Hl.T, sigma=s)
    field = np.divide(Lvl, Cnt, out=np.full_like(Lvl, 2.0), where=Cnt > 1e-6) / 4.0  # 0..1
    dens = Cnt / Cnt.max()
    rgba = STANCE(np.clip(field, 0, 1))
    rgba[..., 3] = np.clip(dens ** 0.42 * 1.15, 0, 0.97)     # 発光は密度比例（空白は黒）

    fig, ax = plt.subplots(figsize=(12.6, 11))
    fig.patch.set_facecolor("#05060a"); ax.set_facecolor("#05060a")
    ax.imshow(rgba, origin="lower", extent=ext, aspect="auto", interpolation="bilinear", zorder=0)

    # 軌跡をごく淡く（流れの質感）
    for i in range(len(meta)):
        c = curves[i]
        ax.plot(c[:, 0], c[:, 1], "-", color=STANCE(lev[i] / 4), lw=0.5, alpha=0.07, zorder=1)

    # 各回答の重心: グロー（大・淡）＋コア（小・明）
    col = STANCE(lev / 4)
    ax.scatter(C[:, 0], C[:, 1], s=240, c=col, alpha=0.10, edgecolors="none", zorder=2)
    ax.scatter(C[:, 0], C[:, 1], s=22, c=col, alpha=0.95, edgecolors="white",
               linewidths=0.3, zorder=3)

    # 5段の重心を輝点として（グロー＋番号）
    cents = np.array([C[lev == k].mean(0) for k in range(5)])
    for k in range(5):
        cx, cy = cents[k]; cc = STANCE(k / 4)
        ax.scatter([cx], [cy], s=1500, color=cc, alpha=0.18, edgecolors="none", zorder=4)
        ax.scatter([cx], [cy], s=300, color=cc, edgecolors="white", linewidths=1.6, zorder=5)
        ax.annotate(f"{k}", (cx, cy), ha="center", va="center", fontsize=12,
                    weight="bold", color="white", zorder=6)
        ax.annotate(LNAME[k], (cx, cy), xytext=(0, 17), textcoords="offset points",
                    ha="center", fontsize=11, weight="bold", color=cc, zorder=6)

    # 立場の連続軸（両端へ矢印）
    t = C @ w; p0 = C[t.argmin()]; p1 = C[t.argmax()]
    ax.add_patch(FancyArrowPatch(tuple(p0), tuple(p1), arrowstyle="-|>", mutation_scale=22,
                 lw=1.4, color="white", alpha=0.5, zorder=4))
    ax.annotate("反対", tuple(p0), xytext=(-6, -16), textcoords="offset points",
                color="#37a0ff", fontsize=12, weight="bold", ha="right")
    ax.annotate("賛成", tuple(p1), xytext=(6, 14), textcoords="offset points",
                color="#ff5a4d", fontsize=12, weight="bold")

    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # カラーバー
    sm = plt.cm.ScalarMappable(cmap=STANCE, norm=plt.Normalize(0, 4)); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.015, ticks=range(5))
    cb.ax.set_yticklabels(LNAME); cb.set_label("立場", color="white")
    cb.ax.yaxis.set_tick_params(color="white"); plt.setp(cb.ax.get_yticklabels(), color="white")
    cb.outline.set_edgecolor("#333")

    ax.set_title("立場は内部状態に『連続』として刻まれる", fontsize=22, weight="bold",
                 color="white", pad=26)
    fig.text(0.5, 0.045, f"原発の是非を反対〜賛成の5段で各40回 — 回答の内部状態をtoorPIAで地図化"
             f"　／　立場の連続性 Spearman = {rho:.2f}",
             ha="center", va="bottom", fontsize=12.5, color="#aeb6c2")

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = os.path.join(RES, "traj_stance_hero.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Spearman={rho:.3f}  saved: {out}")


if __name__ == "__main__":
    main()
