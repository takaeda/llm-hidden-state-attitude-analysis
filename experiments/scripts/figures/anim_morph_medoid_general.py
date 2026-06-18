#!/usr/bin/env python3
"""【汎用】最大3グループの入力プロンプト群を、層モーフ＋各グループの芯(medoid)contour で可視化。

- 比較するグループは最大3（cyan / magenta / yellow のネオン色で区別）。
- medoid は「最終層・高次元(2,560D)空間で DTW により決めた代表回答」を全層で固定して使う。
- 各グループの medoid 出力文を図中に表示（最大3）。
- 出力数 N・タイトル/小見出し/グループ名はすべて可変。座標は morph と同一（整合＋log スケール）。

前提データ: results/<tag>/<outdir>/ に vec_L{n}.npy, xy_L{n}.npy, meta.npz(+ groups,group_names),
            （任意）docs.json（各回答の全文・グループ）。
出力: results/<tag>/anim_<outdir>_medoid.mp4

usage:
  python scripts/figures/anim_morph_medoid_general.py --model Qwen/Qwen3-4B --outdir-name morph3 \
    --title "層が深くなるにつれ各立場が分化していく" --labels "強く反対,中立,強く賛成"
"""
import argparse
import json
import os
import sys
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch, Rectangle
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist

NEON = ["#00e5ff", "#ff2bd6", "#ffe83d"]            # cyan / magenta / yellow
GLOW = ["#9af6ff", "#ff9bef", "#fff39a"]            # medoid 強調（明るめ）


def orth_align(X, Y):
    M = X.T @ Y; U, S, Vt = np.linalg.svd(M); return X @ (U @ Vt)


def field(xy, grp, ext, bins, sigma, cols, gamma=0.5, contrast=1.5, sat=1.4):
    rng = [[ext[0], ext[1]], [ext[2], ext[3]]]
    rgb = np.zeros((bins, bins, 3)); total = np.zeros((bins, bins))
    for g, col in enumerate(cols):
        m = (grp == g)
        if not m.any():
            continue
        H, _, _ = np.histogram2d(xy[m, 0], xy[m, 1], bins=bins, range=rng)
        D = gaussian_filter(H.T, sigma); D /= D.max() + 1e-9
        rgb += np.array(col)[None, None, :] * (D ** gamma)[..., None]; total += D
    g_ = rgb.mean(-1, keepdims=True); rgb = g_ + sat * (rgb - g_)
    rgb = np.clip(rgb * contrast, 0, 1)
    a = np.clip((total / (total.max() + 1e-9)) ** 0.45 * 1.3, 0, 0.99)
    return np.dstack([rgb, a])


def tight_ext(xy, m=1.12):
    lo = np.percentile(xy, 1.0, axis=0); hi = np.percentile(xy, 99.0, axis=0)
    half = max(hi[0] - lo[0], hi[1] - lo[1]) / 2 * m
    cx, cy = (lo + hi) / 2
    return [cx - half, cx + half, cy - half, cy + half]


def dtw_cost(C):
    na, nb = C.shape; D = np.full((na + 1, nb + 1), np.inf); D[0, 0] = 0.0
    for i in range(1, na + 1):
        Ci = C[i - 1]; Dp = D[i - 1]; Dc = D[i]
        for j in range(1, nb + 1):
            Dc[j] = Ci[j - 1] + min(Dp[j], Dc[j - 1], Dp[j - 1])
    return D[na, nb] / (na + nb)


def hd_medoid(chains):
    n = len(chains); M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_cost(cdist(chains[i], chains[j])); M[i, j] = M[j, i] = d
    return int(M.sum(1).argmin())


def short(t, n=104):
    t = " ".join(t.split())
    return (t[:n].rsplit(" ", 1)[0] + " …") if len(t) > n else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--outdir-name", required=True)
    ap.add_argument("--groups", default="", help="表示する group_name をカンマ区切りで（最大3・既定=先頭3）")
    ap.add_argument("--labels", default="", help="図中の表示名（カンマ区切り・グループ順）")
    ap.add_argument("--title", default="層が深くなるにつれ各グループの内部表現が分化していく")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--no-text", action="store_true", help="medoid 出力文を表示しない")
    ap.add_argument("--bins", type=int, default=760)
    ap.add_argument("--sigma", type=float, default=4.5)
    ap.add_argument("--k", type=int, default=11)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dpi", type=int, default=130)
    args = ap.parse_args()
    plt.rcParams["font.family"] = "IPAexGothic"
    plt.rcParams["text.parse_math"] = False

    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tag = args.model.split("/")[-1]
    qdir = os.path.join(base, "results", tag, args.outdir_name)
    meta = np.load(os.path.join(qdir, "meta.npz"), allow_pickle=True)
    sz = meta["sizes"]; bounds = np.cumsum([0] + list(sz))
    gnames = [str(x) for x in meta["group_names"]] if "group_names" in meta else \
        [str(x) for x in sorted(set(int(v) for v in meta["levels"]))]
    gdoc = meta["groups"] if "groups" in meta else meta["levels"]
    layers = sorted(int(x) for x in meta["layers"]
                    if os.path.exists(os.path.join(qdir, f"xy_L{int(x)}.npy")))
    xy = {L: np.load(os.path.join(qdir, f"xy_L{L}.npy")) for L in layers}

    sel = args.groups.split(",") if args.groups else gnames[:3]
    sel = [g for g in sel if g in gnames][:3]
    gid = {g: gnames.index(g) for g in sel}
    G = len(sel)
    labels = args.labels.split(",") if args.labels else sel
    labels = (labels + sel)[:G]
    cols = [to_rgb(NEON[i]) for i in range(G)]
    # per-doc → 表示グループ index（0..G-1）、対象外は -1
    remap = {gid[g]: i for i, g in enumerate(sel)}
    grp = np.array([remap.get(int(gdoc[d]), -1) for d in range(len(sz))])

    # 整合（全トークン・最終層固定・回転/鏡像反転）→ log(隠れ状態RMS)スケール
    desc = sorted(layers, reverse=True)
    A = {desc[0]: xy[desc[0]].copy()}
    for i in range(1, len(desc)):
        A[desc[i]] = orth_align(xy[desc[i]], A[desc[i - 1]])
    Amain = {}
    for L in layers:
        v = np.load(os.path.join(qdir, f"vec_L{L}.npy")).astype(np.float32)
        hr = float(np.sqrt(((v - v.mean(0)) ** 2).sum(1).mean()))
        cur = np.sqrt((A[L] ** 2).sum(1).mean())
        Amain[L] = A[L] / (cur + 1e-9) * np.log(hr)

    # トークン単位の表示グループ（field 用）
    tg = np.concatenate([np.full(int(sz[d]), grp[d]) for d in range(len(sz))])

    # medoid = 最終層・高次元で決定 → 全層で固定
    maxL = max(layers); vF = np.load(os.path.join(qdir, f"vec_L{maxL}.npy")).astype(np.float32)
    medoid_doc = {}
    for gi, g in enumerate(sel):
        docs = [d for d in range(len(sz)) if grp[d] == gi]
        chains = [vF[bounds[d]:bounds[d + 1]] for d in docs]
        mi = hd_medoid(chains)
        medoid_doc[gi] = docs[mi]
        print(f"group '{g}': HD-final medoid = doc#{docs[mi]} (group内 index {mi}, n={len(docs)})")

    # medoid 出力文
    mtexts = {gi: "" for gi in range(G)}
    dp = os.path.join(qdir, "docs.json")
    if os.path.exists(dp) and not args.no_text:
        docs_meta = json.load(open(dp))
        for gi in range(G):
            mtexts[gi] = docs_meta[medoid_doc[gi]]["text"]

    allxy = np.vstack([Amain[L][:, :] for L in layers])
    lo = np.percentile(allxy, 0.7, axis=0); hi = np.percentile(allxy, 99.3, axis=0)
    half = max(hi[0] - lo[0], hi[1] - lo[1]) / 2 * 1.08
    cx, cy = (lo + hi) / 2
    ext = [cx - half, cx + half, cy - half, cy + half]
    asc = sorted(layers); n = len(asc)

    fig = plt.figure(figsize=(13, 11.6)); fig.patch.set_facecolor("#05060a")
    axm = fig.add_axes([0.205, 0.265, 0.59, 0.565]); axm.set_facecolor("#05060a")
    im = axm.imshow(field(Amain[asc[0]], tg, ext, args.bins, args.sigma, cols), origin="lower",
                    extent=ext, aspect="auto", interpolation="bilinear")
    axm.set_xlim(ext[0], ext[1]); axm.set_ylim(ext[2], ext[3]); axm.set_aspect("equal")
    axm.set_xticks([]); axm.set_yticks([])
    for sp in axm.spines.values():
        sp.set_visible(False)
    mlines = []
    for gi in range(G):
        gl, = axm.plot([], [], "-", color=GLOW[gi], lw=3.4, alpha=0.32, zorder=4)
        cw, = axm.plot([], [], "-", color="white", lw=1.5, alpha=0.95, zorder=5)
        tn, = axm.plot([], [], "-", color=GLOW[gi], lw=0.9, alpha=1.0, zorder=6)
        mlines.append((gl, cw, tn))

    fig.text(0.5, 0.975, f"{args.title}  —  {tag}・全トークン・toorPIA（層間整合・log スケール）",
             ha="center", color="white", fontsize=13.5, weight="bold")
    if args.subtitle:
        fig.text(0.5, 0.95, args.subtitle, ha="center", color="#9aa3b2", fontsize=10.5)
    # medoid 出力文（最大3・色分け・控えめ）
    if not args.no_text and any(mtexts.values()):
        fig.text(0.012, 0.935, "代表（medoid）の出力文：", color="#cfd6e2", fontsize=9.5,
                 weight="bold", ha="left", va="top")
        for gi in range(G):
            fig.text(0.012, 0.912 - gi * 0.024, f"● {labels[gi]}:  {short(mtexts[gi])}",
                     color=NEON[gi], fontsize=8.5, ha="left", va="top")
    depth_t = fig.text(0.205, 0.842, "", color="#cfd6e2", fontsize=14, weight="bold", ha="left")
    handles = [Patch(facecolor=NEON[gi], label=labels[gi]) for gi in range(G)]
    handles += [plt.Line2D([0], [0], color="#d8dde6", lw=2, label="各芯 medoid（発光細線）")]
    fig.legend(handles=handles, loc="center", fontsize=10.5, framealpha=0.2, ncol=G + 1,
               labelcolor="white", facecolor="#05060a", edgecolor="#333",
               bbox_to_anchor=(0.5, 0.243))

    # 層リボン
    RX0, RX1, RY0, RH = 0.018, 0.982, 0.055, 0.135
    tw = (RX1 - RX0) / n; pad = tw * 0.10
    for i, L in enumerate(asc):
        axt = fig.add_axes([RX0 + i * tw + pad, RY0, tw - 2 * pad, RH])
        axt.set_facecolor("#05060a")
        te = tight_ext(A[L])
        axt.imshow(field(A[L], tg, te, 170, 2.2, cols), origin="lower", extent=te,
                   aspect="auto", interpolation="bilinear", alpha=0.8)
        axt.set_xlim(te[0], te[1]); axt.set_ylim(te[2], te[3]); axt.set_aspect("equal")
        axt.set_xticks([]); axt.set_yticks([])
        for sp in axt.spines.values():
            sp.set_color("#20242c")
        if L == 1 or L == maxL or L % 5 == 0:
            axt.set_title(f"L{L}", color="#9aa3b2", fontsize=7, pad=1)
    fig.text(0.5, 0.205, "層リボン：　浅 ←                                     層が深くなる方向                                     → 深",
             ha="center", color="#cfd6e2", fontsize=11)
    hl = Rectangle((RX0, RY0 - 0.004), tw, RH + 0.008, transform=fig.transFigure,
                   fill=False, edgecolor="#ffffff", lw=2.4, zorder=10)
    fig.add_artist(hl)

    HOLD = {0: 38, n - 1: 34}
    frames = [(0, 0.0)] * HOLD.get(0, 8)
    for i in range(n - 1):
        for f in range(1, args.k + 1):
            frames.append((i, f / args.k))
    frames += [(n - 1, 0.0)] * HOLD.get(n - 1, 22)
    print(f"{len(frames)} frames")

    def update(idx):
        i, a = frames[idx]; ae = a * a * (3 - 2 * a)
        L0 = asc[i]; L1 = asc[min(i + 1, n - 1)]
        xyf = (1 - ae) * Amain[L0] + ae * Amain[L1]; lv = L0 + ae * (L1 - L0)
        im.set_data(field(xyf, tg, ext, args.bins, args.sigma, cols))
        arts = [im, hl]
        for gi in range(G):
            d = medoid_doc[gi]; c = xyf[bounds[d]:bounds[d + 1]]
            for ln in mlines[gi]:
                ln.set_data(c[:, 0], c[:, 1]); arts.append(ln)
        depth_t.set_text(f"層 L = {int(round(lv))} / {maxL}")
        pos = (lv - asc[0]) / (maxL - asc[0]) * (n - 1); hl.set_x(RX0 + pos * tw)
        return arts

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=1000 / args.fps, blit=False)
    out = os.path.join(base, "results", tag, f"anim_{args.outdir_name}_medoid.mp4")
    anim.save(out, writer=animation.FFMpegWriter(fps=args.fps, bitrate=6500), dpi=args.dpi)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
