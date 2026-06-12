#!/usr/bin/env python3
"""教材図 images/02_spread_two_questions.png を生成する（ADVANCED スライド7）。

st01/sf07 各12試行の出だし3トークンhidden stateを、本編スライド5・6と
同一のtoorPIAマップ(toorpia_map12_full_nonorm_xy.npy)から切り出して表示。
タイトルの散らばり数値は高次元コサイン距離で計算（2次元図に依存しない）。

sf07の構造（全点検証済み）: Yes三重×2 / typically exists×5 /
typically has×2 / 正式名称×3 の4状態。赤(二重らせん派)は言い回しで3小島。

入力: results/Qwen3-4B/full_first3.npz, full_texts.json,
      toorpia_map12_full_nonorm_xy.npy（無ければ先に fig_e03_map.py を実行）

usage: python fig02_spread_two_questions.py
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "IPAexGothic"


def make_square(ax, pts, pad=1.30):
    x, y = pts[:, 0], pts[:, 1]
    cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    half = max(x.max() - x.min(), y.max() - y.min()) / 2 * pad + 1e-9
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")


def spread(vs):
    vn = vs / (np.linalg.norm(vs, axis=1, keepdims=True) + 1e-9)
    S = vn @ vn.T
    return float(1 - S[np.triu_indices(len(vs), 1)].mean())


EXP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(EXP)
RES = os.path.join(EXP, "results", "Qwen3-4B")
OUT = os.path.join(ROOT, "images", "02_spread_two_questions.png")
XY = os.path.join(RES, "toorpia_map12_full_nonorm_xy.npy")

QS = ["st01", "st02", "st08", "sf01", "sf02", "sf07",
      "ct01", "ct04", "ct06", "po01", "po02", "po06"]
if not os.path.exists(XY):
    raise SystemExit("座標キャッシュが無い。先に fig_e03_map.py を実行すること")
xy = np.load(XY)
z = np.load(os.path.join(RES, "full_first3.npz"))
A, B = z["st01"], z["sf07"]
P_st = xy[QS.index("st01") * 12:QS.index("st01") * 12 + 12]
P_sf = xy[QS.index("sf07") * 12:QS.index("sf07") * 12 + 12]

texts = json.load(open(os.path.join(RES, "full_texts.json")))["sf07"]["texts"]
yes = [i for i, t in enumerate(texts) if t.strip().lower().startswith("yes")]
g_exists = [i for i, t in enumerate(texts) if t.startswith("DNA typically exists")]
g_has = [i for i, t in enumerate(texts) if t.startswith("DNA typically has")]
g_paren = [i for i, t in enumerate(texts) if t.startswith("DNA (")]

rng = np.random.default_rng(0)
jit = np.vstack([P_st, P_sf]).std() * 0.02
J = lambda Q, idx: (Q[idx, 0] + rng.normal(0, jit, len(idx)),
                    Q[idx, 1] + rng.normal(0, jit, len(idx)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 6.8), sharex=True, sharey=True)
ax1.scatter(*J(P_st, list(range(12))), s=130, color="#2ca02c", alpha=0.6,
            edgecolors="white", label='12回とも "Yes, the Earth..."')
ax1.set_title(f"「地球は太陽を回る？」\n散らばり = {spread(A):.3f} ― ほぼ一点に重なる（一貫）",
              fontsize=11.5)
ax1.legend(loc="upper left", fontsize=9.5)

ax2.scatter(*J(P_sf, g_exists + g_has + g_paren), s=130, color="#d62728",
            alpha=0.6, edgecolors="white",
            label="「二重らせんだ」と答えた回 ×10（言い回し3系統で3小島）")
ax2.scatter(*J(P_sf, yes), s=150, color="#ff9900", alpha=0.85,
            edgecolors="white", marker="^",
            label=f"「Yes, 三重らせんもある」と答えた回 ×{len(yes)}")
ax2.set_title(f"「DNAは三重らせん？」\n散らばり = {spread(B):.3f} ― 立場・言い回しごとの塊に割れる（非一貫）",
              fontsize=11.5)
ax2.legend(loc="upper left", fontsize=9.5)
for idx, lab, dxy in [
        (g_exists, f"exists系 ×{len(g_exists)}", (-45, 30)),
        (g_has, f"has系 ×{len(g_has)}", (30, 60)),
        (g_paren, f"正式名称系 ×{len(g_paren)}", (60, 25))]:
    ax2.annotate(lab, xy=(P_sf[idx, 0].mean(), P_sf[idx, 1].mean()),
                 xytext=dxy, textcoords="offset points", fontsize=9.5,
                 color="#a33", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#a33"))

for ax in (ax1, ax2):
    ax.set_xlabel("toorPIAによる2次元化（規格化なし・本編の地図と同一マップ）")
    ax.grid(alpha=0.3)
    make_square(ax, np.vstack([P_st, P_sf]))
fig.suptitle("同じ質問に12回答えさせ、各回の「出だし数語の hidden state」を1点として描く（Qwen3-4B 実測）",
             fontsize=12.5, weight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(OUT, dpi=140)
print(f"saved: {OUT}")
