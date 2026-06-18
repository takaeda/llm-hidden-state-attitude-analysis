#!/usr/bin/env python3
"""層別 vec_L{n}.npy を toorPIA でマップ化し xy_L{n}.npy を保存（再開可能）。

各層を fit_transform_csvform で2次元化。既に xy_L{n}.npy がある層はスキップするので、
途中で落ちても再実行で続きから。全層が同一生成系列由来なのでキルト図は層間で比較可能。

事前（大規模 fit のため必須）:
  source ~/work/toorpia/samples/env.sh
  export TOORPIA_API_URL=http://localhost:3000   # 本文上限の高いローカル直結
usage:
  python scripts/16b_fit_layers_toorpia.py --model Qwen/Qwen3-4B
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"))
from fig_traj_dna import make_basemap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--vnorm", action="store_true", help="vector_normalization=True で fit")
    ap.add_argument("--layers", default="", help="カンマ区切りで層を限定（省略時 meta.npz の全層）")
    ap.add_argument("--outdir-name", default="quilt", help="results/<tag>/ 下の対象フォルダ名")
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tag = args.model.split("/")[-1]
    outdir = os.path.join(base, "results", tag, args.outdir_name)
    meta = np.load(os.path.join(outdir, "meta.npz"))
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else [int(x) for x in meta["layers"]])

    for L in layers:
        vecp = os.path.join(outdir, f"vec_L{L}.npy")
        xyp = os.path.join(outdir, f"xy_L{L}.npy")
        if os.path.exists(xyp):
            print(f"L{L}: 既存 xy をスキップ")
            continue
        if not os.path.exists(vecp):
            print(f"L{L}: vec_L{L}.npy が無い。スキップ"); continue
        X = np.load(vecp).astype(np.float32)
        print(f"L{L}: fit {X.shape} ...")
        make_basemap(X, xyp, vnorm=args.vnorm, recompute=True)  # 内部で xyp に保存
        print(f"L{L}: saved {xyp}")
    print("done.")


if __name__ == "__main__":
    main()
