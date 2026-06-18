#!/usr/bin/env python3
"""各グループの『medoid 周りの揺らぎ（束の広がり）』を層別に定量し比較する。

温度を揃えた生成(同条件)前提。各層の toorPIA マップ上で、各グループの固定medoid(最終層HD-DTW)から
他N-1本への平均DTW距離を求め、その層の全体スケール(全点の重心RMS)で正規化＝『相対揺らぎ』。
これを層×グループでプロット（neon: cyan/magenta/yellow）。

入力: results/<tag>/<outdir>/xy_L{n}.npy, vec_L{n}.npy, meta.npz, (任意)docs.json
出力: results/<tag>/<outdir>_bundle_spread.png, .json
usage: python scripts/figures/fig_bundle_spread.py --model Qwen/Qwen3-4B --outdir-name morph3 --labels "強く反対,中立,強く賛成"
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

NEON = ["#00b8d4", "#d400b0", "#c9a200"]   # cyan/magenta/yellow（線用に少し濃いめ）


def dtw_cost(C):
    na, nb = C.shape; D = np.full((na + 1, nb + 1), np.inf); D[0, 0] = 0.0
    for i in range(1, na + 1):
        Ci = C[i - 1]; Dp = D[i - 1]; Dc = D[i]
        for j in range(1, nb + 1):
            Dc[j] = Ci[j - 1] + min(Dp[j], Dc[j - 1], Dp[j - 1])
    return D[na, nb] / (na + nb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--outdir-name", required=True)
    ap.add_argument("--labels", default="")
    args = ap.parse_args()
    plt.rcParams["font.family"] = "IPAexGothic"
    plt.rcParams["text.parse_math"] = False

    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tag = args.model.split("/")[-1]
    qdir = os.path.join(base, "results", tag, args.outdir_name)
    meta = np.load(os.path.join(qdir, "meta.npz"), allow_pickle=True)
    sz = meta["sizes"]; bounds = np.cumsum([0] + list(sz))
    gdoc = meta["groups"] if "groups" in meta else meta["levels"]
    gnames = [str(x) for x in meta["group_names"]] if "group_names" in meta else \
        [str(g) for g in sorted(set(int(v) for v in gdoc))]
    layers = sorted(int(x) for x in meta["layers"]
                    if os.path.exists(os.path.join(qdir, f"xy_L{int(x)}.npy")))
    G = len(gnames)
    labels = args.labels.split(",") if args.labels else gnames
    docs_by = {g: [d for d in range(len(sz)) if int(gdoc[d]) == g] for g in range(G)}

    # medoid = 最終層・高次元
    maxL = max(layers); vF = np.load(os.path.join(qdir, f"vec_L{maxL}.npy")).astype(np.float32)
    med = {}
    for g in range(G):
        ch = [vF[bounds[d]:bounds[d + 1]] for d in docs_by[g]]
        n = len(ch); M = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = dtw_cost(cdist(ch[i], ch[j])); M[i, j] = M[j, i] = d
        med[g] = docs_by[g][int(M.sum(1).argmin())]

    # 各層: medoid から他への平均DTW（その層の全体RMSで正規化）
    rel = {g: [] for g in range(G)}
    for L in layers:
        xy = np.load(os.path.join(qdir, f"xy_L{L}.npy"))
        S = np.sqrt(((xy - xy.mean(0)) ** 2).sum(1).mean()) + 1e-9   # 層の全体スケール
        for g in range(G):
            mc = xy[bounds[med[g]]:bounds[med[g] + 1]]
            ds = [dtw_cost(cdist(xy[bounds[d]:bounds[d + 1]], mc))
                  for d in docs_by[g] if d != med[g]]
            rel[g].append(float(np.mean(ds)) / S)

    Ls = np.array(layers)
    fig, ax = plt.subplots(figsize=(11, 6.2))
    means = {}
    for g in range(G):
        ax.plot(Ls, rel[g], "-o", color=NEON[g % 3], lw=2.3, ms=4.5, label=labels[g])
        means[g] = float(np.mean(rel[g]))
    ax.set_xlabel("層（hidden_states 索引 / 36=最終層）", fontsize=12)
    ax.set_ylabel("medoid周りの相対揺らぎ（束の平均DTW距離 ÷ 層スケール）", fontsize=11.5)
    ax.set_title(f"medoid周りの『揺らぎ』の層プロファイル（同temperature・{tag}）\n"
                 + " ／ ".join(f"{labels[g]}平均={means[g]:.3f}" for g in range(G)),
                 fontsize=12.5, weight="bold")
    ax.set_xticks(Ls[::2]); ax.grid(alpha=0.25); ax.legend(fontsize=11, loc="best")
    fig.tight_layout()
    out = os.path.join(base, "results", tag, f"{args.outdir_name}_bundle_spread.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    json.dump({"labels": labels, "layers": layers, "rel_spread": rel, "means": means,
               "medoid_doc": med}, open(os.path.join(base, "results", tag,
               f"{args.outdir_name}_bundle_spread.json"), "w"), ensure_ascii=False, indent=1)
    print("means:", {labels[g]: round(means[g], 3) for g in range(G)})
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
