#!/usr/bin/env python3
"""Qwen3-4B に論争質問2つ（AIは仕事を奪うか / 原発は正しい政策か）を各100回・簡潔回答。
全トークン(smooth5)を1枚の共有 toorPIA マップに投入し、トピックで色分け。

確認点:
 (a) 各トピック内は意味的に1パターンに収束（割れない）→ 1クラスタに集まる
 (b) 異なるトピック同士は内容が遠い → マップ上で分離する（内容距離で分ける、の正対照）

事前(fit時): source ~/work/toorpia/samples/env.sh; export TOORPIA_API_URL=http://localhost:3000
usage: python figures/fig_traj_contested.py [--refit]
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_traj_dna import make_basemap, HEAT_CMAP  # noqa: E402
from fig_traj import moving_average  # noqa: E402

plt.rcParams["font.family"] = "IPAexGothic"
plt.rcParams["text.parse_math"] = False
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "results", "Qwen3-4B")
JA = {"ai_jobs": "AIは仕事を奪うか", "nuclear": "原発は正しい政策か"}
COL = {"ai_jobs": "#2fe0e0", "nuclear": "#ff5fd0"}


def load_two(csvp):
    """contested CSV を1回読み、(sent_id,variant)ごとに生成トークン(dカラム)を返す。"""
    import csv as _csv
    _csv.field_size_limit(10 ** 7)
    rows = {}
    with open(csvp, newline="") as f:
        r = _csv.reader(f); hdr = next(r)
        idx = {c: i for i, c in enumerate(hdr)}
        dcol = [i for i, c in enumerate(hdr) if c.startswith("d")]
        si, vi, ti, ai = idx["sent_id"], idx["variant"], idx["tok_idx"], idx["is_anchor"]
        for row in r:
            if row[ai] == "1":
                continue
            key = (row[si], int(row[vi]))
            rows.setdefault(key, []).append((int(row[ti]), [float(row[i]) for i in dcol]))
    out = {}
    for k, lst in rows.items():
        lst.sort(key=lambda x: x[0])
        out[k] = np.array([x[1] for x in lst])
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--refit", action="store_true")
    args = ap.parse_args()
    meta = json.load(open(os.path.join(RES, "traj_meta_contested.json")))
    cache = os.path.join(RES, "traj_contested_xy.npy")
    # 文書順を確定（meta順）
    keys = [(m["sent_id"], m["variant"]) for m in meta]
    topic = [m["sent_id"] for m in meta]
    if args.refit or not os.path.exists(cache):
        allX = load_two(os.path.join(RES, "traj_vectors_contested.csv"))
        segs = [moving_average(allX[k], 5) for k in keys]
        XY = make_basemap(np.vstack(segs), cache, recompute=True)
        np.save(os.path.join(RES, "traj_contested_sizes.npy"),
                np.array([len(s) for s in segs]))
    else:
        XY = np.load(cache)
    sizes = np.load(os.path.join(RES, "traj_contested_sizes.npy"))
    bounds = np.cumsum([0] + list(sizes))
    assert bounds[-1] == len(XY), (bounds[-1], len(XY))
    curves = [XY[bounds[i]:bounds[i + 1]] for i in range(len(meta))]
    print("docs:", dict(Counter(topic)))

    # トピック分離（教師なしk=2 vs トピック）
    C = np.array([c.mean(0) for c in curves])
    true = np.array([0 if t == "ai_jobs" else 1 for t in topic])
    km = KMeans(2, n_init=20, random_state=0).fit_predict(C)
    ari = adjusted_rand_score(true, km)
    cA = C[true == 0].mean(0); cB = C[true == 1].mean(0)
    win = np.mean([np.linalg.norm(C[i] - (cA if true[i] == 0 else cB)) for i in range(len(C))])
    sep = float(np.linalg.norm(cA - cB) / (win + 1e-9))
    print(f"トピック分離: 教師なしARI={ari:.3f}  分離比={sep:.2f}  混同={confusion_matrix(true,km).tolist()}")

    # 図
    fig, ax = plt.subplots(figsize=(11.5, 10))
    fig.patch.set_facecolor("black"); ax.set_facecolor("black")
    xmin, ymin = XY.min(0); xmax, ymax = XY.max(0)
    mx, my = (xmax - xmin) * .05, (ymax - ymin) * .05
    ext = [xmin - mx, xmax + mx, ymin - my, ymax + my]
    H, _, _ = np.histogram2d(XY[:, 0], XY[:, 1], bins=440, range=[[ext[0], ext[1]], [ext[2], ext[3]]])
    Zn = gaussian_filter(H.T, sigma=8); Zn /= Zn.max()
    rgba = HEAT_CMAP(np.clip(Zn ** 0.7, 0, 0.82)); rgba[..., 3] = np.clip(Zn ** 0.35, 0, 0.9)
    ax.imshow(rgba, origin="lower", extent=ext, aspect="auto", interpolation="bilinear", zorder=0)
    for i in range(len(meta)):
        c = curves[i]
        ax.plot(c[:, 0], c[:, 1], "-", color=COL[topic[i]], lw=0.7, alpha=0.3, zorder=2)
    for i in range(len(meta)):
        ax.scatter([curves[i][0, 0]], [curves[i][0, 1]], marker="o", s=20,
                   color=COL[topic[i]], edgecolors="white", linewidths=0.3, alpha=0.9, zorder=4)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    cnt = Counter(topic)
    handles = [Line2D([0], [0], color=COL[k], lw=3, label=f"{JA[k]} n={cnt[k]}") for k in JA]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
              fontsize=11, frameon=True, facecolor="black", edgecolor="white", labelcolor="white")
    ax.set_title("論争質問2つ × 各100回（Qwen3-4B・簡潔回答）を1枚の共有マップに\n"
                 f"全{len(XY)}点の密度ヒートマップ + トピック別軌跡（toorPIA）  "
                 f"トピック分離 ARI={ari:.2f}・分離比={sep:.1f}",
                 fontsize=12, weight="bold", color="white")
    fig.tight_layout()
    out = os.path.join(RES, "traj_contested.png")
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="black")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
