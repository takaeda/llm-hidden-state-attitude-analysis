#!/usr/bin/env python3
"""1文章の hidden state 軌跡を toorPIA マップ上に描き、形を定量化する（PLAN3）。

traj_vectors.csv（15_trajectory.py 出力）から1文章ぶんの全トークンを取り出し、
toorPIA に投入して2D座標を得て、生成順に線で結ぶ。色=生成順（時間）。
直進度・総回転角・自己交差・閉合度などを toorPIA-2D / PCA-2D / 元次元 で出し、
棒状/ループ状/トグロ状を自動判定する。

事前に: source ~/work/toorpia/samples/env.sh

usage:
  python figures/fig_traj.py --sent-id p1_earth                 # greedy v0, norm=False
  python figures/fig_traj.py --sent-id p3_nuclear --norm        # vector_normalization=True
  python figures/fig_traj.py --sent-id p5_repeat --recompute    # キャッシュ無視で再fit
"""
import argparse
import csv
import json
import os

import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
# tight_layout の非互換警告だけ黙らせる。フォント欠落(=文字化け)の警告は残す。
warnings.filterwarnings("ignore", message=".*tight_layout.*")
warnings.filterwarnings("ignore", message=".*Tight layout not applied.*")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

plt.rcParams["font.family"] = "IPAexGothic"
plt.rcParams["text.parse_math"] = False  # 生成文中の $ や ** を数式扱いしない
csv.field_size_limit(10 ** 7)

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------- データ読み込み ----------
def load_sentence(csv_path, sent_id, gen_mode, variant, with_anchor):
    meta, X = [], []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        dcols = [c for c in r.fieldnames if c.startswith("d")]
        for row in r:
            if (row["sent_id"] == sent_id and row["gen_mode"] == gen_mode
                    and int(row["variant"]) == variant):
                ti = int(row["tok_idx"])
                if ti == -1 and not with_anchor:
                    continue
                meta.append({"tok_idx": ti, "token": row["token"],
                             "is_boundary": int(row["is_boundary"]),
                             "is_anchor": int(row["is_anchor"]),
                             "norm": float(row["norm"])})
                X.append([float(row[c]) for c in dcols])
    order = np.argsort([m["tok_idx"] for m in meta])
    meta = [meta[i] for i in order]
    X = np.asarray(X, dtype=float)[order]
    return meta, X


# ---------- 幾何指標 ----------
def pca2(X):
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def moving_average(X, k):
    """高次元のまま中心移動平均。トークン固有の跳ねを均し、粗いドリフトを出す。"""
    if k <= 1:
        return X
    n, h = len(X), k // 2
    out = np.empty_like(X)
    for i in range(n):
        out[i] = X[max(0, i - h):min(n, i + h + 1)].mean(0)
    return out


def seg_intersect(p1, p2, p3, p4):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = ccw(p3, p4, p1), ccw(p3, p4, p2)
    d3, d4 = ccw(p1, p2, p3), ccw(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def count_self_intersections(P):
    n = len(P) - 1
    c = 0
    for i in range(n):
        for j in range(i + 2, n):
            if seg_intersect(P[i], P[i + 1], P[j], P[j + 1]):
                c += 1
    return c


def metrics_nd(X):
    step = np.linalg.norm(np.diff(X, axis=0), axis=1)
    L = float(step.sum())
    D = float(np.linalg.norm(X[-1] - X[0]))
    return {"path_len": round(L, 4), "end_disp": round(D, 4),
            "straightness": round(D / L if L else 0.0, 4)}


def metrics_2d(P):
    diffs = np.diff(P, axis=0)
    step = np.linalg.norm(diffs, axis=1)
    L = float(step.sum())
    D = float(np.linalg.norm(P[-1] - P[0]))
    ang = np.arctan2(diffs[:, 1], diffs[:, 0])
    dang = np.diff(ang)
    dang = (dang + np.pi) % (2 * np.pi) - np.pi
    total_turn = float(np.abs(dang).sum())
    winding = float(dang.sum())
    nsi = count_self_intersections(P)
    N = len(P)
    q = max(1, N // 4)
    dd = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    diam = float(dd.max())
    closure = float(np.linalg.norm(P[-q:] - P[0], axis=1).min() / (diam + 1e-9))
    C = P - P.mean(0)
    ev = np.sort(np.linalg.eigvalsh(C.T @ C / len(P)))[::-1]
    gyr = float(np.sqrt(ev.sum()))
    aspect = float(np.sqrt(ev[0] / (ev[1] + 1e-12)))
    return {"path_len": round(L, 4), "end_disp": round(D, 4),
            "straightness": round(D / L if L else 0.0, 4),
            "total_turn_rad": round(total_turn, 3),
            "winding_turns": round(total_turn / (2 * np.pi), 3),
            "n_self_int": nsi, "closure": round(closure, 4),
            "radius_gyration": round(gyr, 4), "aspect_ratio": round(aspect, 3),
            "step_mean": round(float(step.mean()), 4),
            "step_cv": round(float(step.std() / (step.mean() + 1e-9)), 3)}


def shape_label(m):
    turns = m["winding_turns"]
    if m["straightness"] >= 0.7 and m["n_self_int"] == 0:
        return "棒状 (rod)"
    if m["closure"] <= 0.15 and m["n_self_int"] >= 1 and turns < 1.5:
        return "ループ状 (loop)"
    if turns >= 1.5 and m["n_self_int"] >= 2:
        return "トグロ状 (coil)"
    return "中間/不定 (mixed)"


# ---------- toorPIA ----------
def toorpia_xy(X, cache, recompute, vnorm):
    if os.path.exists(cache) and not recompute:
        return np.load(cache)
    from toorpia.client import toorPIA
    tmp = "/tmp/traj_fit.csv"
    with open(tmp, "w") as f:
        f.write("rid," + ",".join(f"d{i}" for i in range(X.shape[1])) + "\n")
        for k in range(len(X)):
            f.write(f"{k}," + ",".join(f"{v:.5f}" for v in X[k]) + "\n")
    client = toorPIA()
    xy = np.asarray(client.fit_transform_csvform(
        tmp, drop_columns=["rid"], label="hidden-state trajectory",
        tag="trajectory", vector_normalization=vnorm), dtype=float)
    np.save(cache, xy)
    return xy


# ---------- 描画 ----------
def draw(sid, mode, v, meta, xy, m_toor, m_pca, m_full, label, text, outpng, vnorm):
    tok_idx = np.array([m["tok_idx"] for m in meta])
    norms = np.array([m["norm"] for m in meta])
    bnd = np.array([m["is_boundary"] for m in meta], dtype=bool)
    N = len(xy)
    cidx = np.arange(N)

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.1, 1], height_ratios=[1, 1])
    ax = fig.add_subplot(gs[:, 0])
    axn = fig.add_subplot(gs[0, 1])
    axs = fig.add_subplot(gs[1, 1])

    pts = xy.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="viridis", norm=Normalize(0, N - 1), zorder=2)
    lc.set_array(cidx[:-1])
    lc.set_linewidth(2.2)
    ax.add_collection(lc)
    ax.scatter(xy[:, 0], xy[:, 1], c=cidx, cmap="viridis", s=18,
               zorder=3, edgecolors="none")
    if bnd.any():
        ax.scatter(xy[bnd, 0], xy[bnd, 1], marker="P", s=70, facecolors="none",
                   edgecolors="black", linewidths=1.1, zorder=4,
                   label="節目(句読点/改行)")
    ax.scatter([xy[0, 0]], [xy[0, 1]], marker="o", s=160, facecolors="none",
               edgecolors="#1f77b4", linewidths=2.4, zorder=5, label="始点")
    ax.scatter([xy[-1, 0]], [xy[-1, 1]], marker="^", s=150, color="#d62728",
               zorder=5, label="終点")
    cb = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("生成順（トークン位置）")
    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale()
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_title(f"hidden state 軌跡: {sid} ({mode} v{v}) — toorPIA"
                 f"（vector_normalization={vnorm}）\n判定: {label}",
                 fontsize=12, weight="bold")

    info = (f"[toorPIA-2D]\n"
            f" 直進度 D/L={m_toor['straightness']}  巻き数={m_toor['winding_turns']}\n"
            f" 自己交差={m_toor['n_self_int']}  閉合度={m_toor['closure']}\n"
            f" アスペクト={m_toor['aspect_ratio']}  step変動={m_toor['step_cv']}\n"
            f"[PCA-2D] 直進度={m_pca['straightness']} 巻き数={m_pca['winding_turns']} 交差={m_pca['n_self_int']}\n"
            f"[元次元(2560)] 直進度={m_full['straightness']}")
    ax.text(0.015, 0.015, info, transform=ax.transAxes, fontsize=9.5,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.92))

    axn.plot(tok_idx, norms, "-o", ms=3, color="#555")
    axn.set_title("ノルム |h|（外れトークン検出）", fontsize=10)
    axn.set_xlabel("tok_idx"); axn.grid(alpha=0.3)

    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    axs.plot(tok_idx[1:], step, "-o", ms=3, color="#2ca02c")
    axs.set_title("マップ上のステップ幅（隣接距離）", fontsize=10)
    axs.set_xlabel("tok_idx"); axs.grid(alpha=0.3)

    fig.suptitle(text[:110] + ("…" if len(text) > 110 else ""),
                 y=0.005, va="bottom", fontsize=9, color="#333")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(outpng, dpi=140)
    print(f"saved: {outpng}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-4B")
    ap.add_argument("--sent-id", required=True)
    ap.add_argument("--gen-mode", default="greedy")
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--norm", action="store_true",
                    help="vector_normalization=True（既定はFalse）")
    ap.add_argument("--with-anchor", action="store_true",
                    help="生成直前のプロンプト末尾を始点として含める")
    ap.add_argument("--suffix", default="",
                    help="読み込む traj_vectors{suffix}.csv の接尾辞（層別。例: _L18）")
    ap.add_argument("--smooth", type=int, default=0,
                    help="高次元での中心移動平均の窓幅（0/1=なし。例: 5）")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    res = os.path.join(BASE, "results", args.model)
    csv_path = os.path.join(res, f"traj_vectors{args.suffix}.csv")
    meta, X = load_sentence(csv_path, args.sent_id, args.gen_mode,
                            args.variant, args.with_anchor)
    if len(X) < 4:
        raise SystemExit(f"トークンが少なすぎ（{len(X)}点）。判定対象外: {args.sent_id}")
    if args.smooth > 1:
        X = moving_average(X, args.smooth)

    vnorm = bool(args.norm)
    stag = f"{args.suffix}_s{args.smooth}" if args.smooth > 1 else args.suffix
    ntag = ("norm" if vnorm else "nonorm") + stag
    cache = os.path.join(res, f"traj_xy_{args.sent_id}_{args.gen_mode}_v{args.variant}_{ntag}.npy")
    xy = toorpia_xy(X, cache, args.recompute, vnorm)

    m_toor = metrics_2d(xy)
    m_pca = metrics_2d(pca2(X))
    m_full = metrics_nd(X)
    label = shape_label(m_toor)

    metas = json.load(open(os.path.join(res, f"traj_meta{args.suffix}.json")))
    rec = next((r for r in metas if r["sent_id"] == args.sent_id
                and r["gen_mode"] == args.gen_mode and r["variant"] == args.variant), {})
    text = rec.get("text", "")

    out = json.load(open(os.path.join(res, "traj_metrics.json"))) \
        if os.path.exists(os.path.join(res, "traj_metrics.json")) else {}
    key = f"{args.sent_id}_{args.gen_mode}_v{args.variant}_{ntag}"
    out[key] = {"label": label, "n_tokens": len(X),
                "toorpia": m_toor, "pca": m_pca, "full": m_full}
    json.dump(out, open(os.path.join(res, "traj_metrics.json"), "w"),
              ensure_ascii=False, indent=1)

    outpng = os.path.join(res, f"traj_{args.sent_id}_{args.gen_mode}_v{args.variant}_{ntag}.png")
    draw(args.sent_id, args.gen_mode, args.variant, meta, xy,
         m_toor, m_pca, m_full, label, text, outpng, vnorm)
    print(f"\n{key}: {label}")
    print(f"  toorPIA: {m_toor}")
    print(f"  PCA    : {m_pca}")
    print(f"  full   : {m_full}")


if __name__ == "__main__":
    main()
