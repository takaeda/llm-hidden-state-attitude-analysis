#!/usr/bin/env python3
"""既知の割れケース sf07「DNAは三重らせん?」を20回サンプリングし、各回答の全トークン
(smooth5)を1枚の共有 toorPIA マップに投入。同一文書を線で結び、回答パターンで色分けして
「回答の割れが軌跡(とくに出だし)にも出るか」を見る。

回答パターン:
  宣言:二重らせん(正) … "DNA typically has a double helix..."
  Yes:三重(誤)        … "Yes, DNA can have a triple helix..."
  明示No:二重(正)      … "DNA does not ..."

事前に: source ~/work/toorpia/samples/env.sh
usage: python figures/fig_traj_dna.py
"""
import json
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde  # noqa: F401 (互換のため残置)
from scipy.ndimage import gaussian_filter

# 濃青→赤→オレンジ→白（低密度→高密度）
HEAT_CMAP = LinearSegmentedColormap.from_list("dblue_red_orange_white", [
    (0.00, "#0a0a4d"), (0.40, "#e01414"), (0.72, "#ff9a00"), (1.00, "#ffffff")])

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_traj import BASE, load_sentence, moving_average  # noqa: E402

plt.rcParams["font.family"] = "IPAexGothic"
plt.rcParams["text.parse_math"] = False
RES = os.path.join(BASE, "results", "Qwen3-4B")

PAT = {"double": ("宣言:二重らせん(正)", "#1f77b4"),
       "yes_triple": ("Yes:三重(誤)", "#d62728"),
       "no_explicit": ("明示No:二重(正)", "#2ca02c"),
       "other": ("その他", "#999999")}


def pattern(text):
    low = re.sub(r"[*#`_]", "", text).strip().lower()   # markdown除去
    if low.startswith("yes"):
        return "yes_triple"
    head = low[:80]
    if "does not" in head or "doesn't" in head or "not typically" in head \
            or head.startswith("no"):
        return "no_explicit"
    if "double helix" in low[:120]:
        return "double"
    return "other"


def load_all_variants(csvp, sent_id="sf07_dna"):
    """CSVを1回だけ読み、変種ごとの生成トークン(dカラム)を tok_idx 昇順で返す。
    （load_sentence を変種数だけ呼ぶとフルCSVを何度も読んで遅いため、一括化）。"""
    import csv as _csv
    _csv.field_size_limit(10 ** 7)
    by_var = {}
    with open(csvp) as f:
        r = _csv.reader(f)
        hdr = next(r)
        idx = {c: i for i, c in enumerate(hdr)}
        dcol = [i for i, c in enumerate(hdr) if c.startswith("d")]
        vi, ti, ai, si = idx["variant"], idx["tok_idx"], idx["is_anchor"], idx["sent_id"]
        for row in r:
            if row[si] != sent_id or row[ai] == "1":
                continue
            by_var.setdefault(int(row[vi]), []).append(
                (int(row[ti]), [float(row[i]) for i in dcol]))
    out = {}
    for v, lst in by_var.items():
        lst.sort(key=lambda x: x[0])
        out[v] = np.array([x[1] for x in lst])
    return out


def make_basemap(X, cache, vnorm=False, recompute=False):
    """toorPIA でベースマップを作り全点の2D座標を返す（fit_transform_csvform を1回呼ぶだけ）。
    アップロードは multipart のCSV。巨大すぎるとAPIに届かず None が返るので呼び出し側で行数を抑える。"""
    if os.path.exists(cache) and not recompute:
        return np.load(cache)
    from toorpia.client import toorPIA
    tmp = "/tmp/traj_dna_fit.csv"
    with open(tmp, "w") as f:
        f.write("rid," + ",".join(f"d{i}" for i in range(X.shape[1])) + "\n")
        for k in range(len(X)):
            f.write(f"{k}," + ",".join(f"{v:.5f}" for v in X[k]) + "\n")
    print(f"  uploading {len(X)} rows ({os.path.getsize(tmp)/1e6:.1f} MB) to toorPIA...")
    xy = toorPIA().fit_transform_csvform(
        tmp, drop_columns=["rid"], label="sf07 DNA trajectories",
        tag="dna-split", vector_normalization=vnorm)
    if xy is None:
        raise SystemExit("toorPIA が None を返しました（アップロード失敗。行数/サイズ超過の可能性）。")
    xy = np.asarray(xy, dtype=float)
    np.save(cache, xy)
    return xy


def main():
    import argparse
    from collections import Counter
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", default="traj_dna_xy.npy")  # 2D座標npy（toorPIA/PCA/t-SNE）
    ap.add_argument("--out", default="traj_dna.png")
    ap.add_argument("--label", default="toorPIA")           # タイトルに出す手法名
    args = ap.parse_args()
    meta = json.load(open(os.path.join(RES, "traj_meta_dna.json")))
    XY = np.load(os.path.join(RES, args.coords))            # 全点の2D座標
    bounds = np.cumsum([0] + [m["n_tokens"] for m in meta])
    assert bounds[-1] == len(XY), (bounds[-1], len(XY))
    curves = [XY[bounds[i]:bounds[i + 1]] for i in range(len(meta))]
    pat_of = {i: pattern(meta[i]["text"]) for i in range(len(meta))}
    cnt = Counter(pat_of.values())
    n = len(curves)
    print(f"{n}回 / 全{len(XY)}点  カウント:", {PAT[k][0]: cnt[k] for k in cnt})

    # === 黒背景 + 全9600点の密度ヒートマップ(jet) + 色付き実線 ===
    fig, ax = plt.subplots(figsize=(11.5, 10))
    fig.patch.set_facecolor("black"); ax.set_facecolor("black")

    xmin, ymin = XY.min(0); xmax, ymax = XY.max(0)
    mx, my = (xmax - xmin) * 0.05, (ymax - ymin) * 0.05
    ext = [xmin - mx, xmax + mx, ymin - my, ymax + my]
    BINS = 440   # メッシュ ×2（旧220→440）
    H, _, _ = np.histogram2d(XY[:, 0], XY[:, 1], bins=BINS,
                             range=[[ext[0], ext[1]], [ext[2], ext[3]]])
    Z = gaussian_filter(H.T, sigma=8)            # 全点の密度。広めに（sigma↑）・等高線なし
    Zn = Z / Z.max()
    GAMMA, TOPCAP = 0.7, 0.82    # 色は緩めのガンマで青→橙の階調を保持 / TOPCAP<1=純白に到達させず白飛び抑制
    Zc = np.clip(Zn ** GAMMA, 0, TOPCAP)
    rgba = HEAT_CMAP(Zc)                            # 低密度=濃青→高密度=赤→橙→(淡)白
    rgba[..., 3] = np.clip(Zn ** 0.35, 0, 0.9)      # 可視性(不透明度)で低密度を持ち上げる（黒地は残す）
    ax.imshow(rgba, origin="lower", extent=ext, aspect="auto",
              interpolation="bilinear", zorder=0)

    # 軌跡は全件1色（意味的に同一の回答=通常二重・条件下で三重も可。表層の二重/三重は区別しない）
    LINE_COL = "#67d8ef"
    for v in range(n):
        c = curves[v]
        ax.plot(c[:, 0], c[:, 1], "-", color=LINE_COL, lw=0.6, alpha=0.22, zorder=2)
    for v in range(n):
        c = curves[v]
        ax.scatter([c[0, 0]], [c[0, 1]], marker="o", s=18, color=LINE_COL,
                   edgecolors="white", linewidths=0.3, alpha=0.85, zorder=4)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    # 凡例（プロット下＝周辺・白文字）
    handles = [Line2D([0], [0], color=LINE_COL, lw=3,
               label=f"回答軌跡（全{n}件・意味はほぼ同一: 通常は二重らせん／条件下で三重も可）")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=1, fontsize=10, frameon=True, facecolor="black",
              edgecolor="white", labelcolor="white")
    # 密度カラーバー(jet)
    sm = plt.cm.ScalarMappable(cmap=HEAT_CMAP, norm=plt.Normalize(0, 1)); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("点密度（全9600点・規格化）", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.get_yticklabels(), color="white")

    ax.set_title(f"sf07「Does DNA have a triple helix?」を{n}回\n"
                 f"全{len(XY)}点の密度ヒートマップ(濃青→赤→橙→白/黒背景) + 回答軌跡（{args.label}・最終層）",
                 fontsize=12, weight="bold", color="white")
    fig.tight_layout()
    out = os.path.join(RES, args.out)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="black")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
