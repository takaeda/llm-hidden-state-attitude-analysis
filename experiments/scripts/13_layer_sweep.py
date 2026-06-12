#!/usr/bin/env python3
"""層スイープ: 「意味・立場」はどの層で最も純粋に表現されるか。

既存の回答群を全層で再forwardし、各層の全文平均hidden stateについて
  立場分離度(ℓ) = 立場間の重心距離 ÷ 言い回し内のブレ（同一立場内の散らばり）
を計算する（比なので層ごとのスケール・異方性は相殺される）。

素材:
  - sf07 (DNA三重らせん):  立場が本当に割れる（Yes×2 / 二重らせん×10）
  - po06 (中国は民主主義): 立場は同一・言い回しのみ多様（純粋な言い回しノイズの指標）
  - complex_sales:          オープンエンド課題（時系列×13 / 顧客分析×1）

usage: python 13_layer_sweep.py --load-4bit
"""
import argparse
import json
import os

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def cosd(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return 1.0 - float(a @ b)


def spread(vs):
    if len(vs) < 2:
        return 0.0
    vn = vs / (np.linalg.norm(vs, axis=1, keepdims=True) + 1e-9)
    S = vn @ vn.T
    return float(1 - S[np.triu_indices(len(vs), 1)].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--load-4bit", action="store_true")
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res = os.path.join(base, "results", args.model.split("/")[-1])
    qall = yaml.safe_load(open(os.path.join(base, "questions_v1.yaml")))
    qmap = {q["id"]: q["text"] for c in qall for q in qall[c]}
    full = json.load(open(os.path.join(res, "full_texts.json")))
    cpx = json.load(open(os.path.join(res, "complex_task.json")))

    # 素材と立場ラベル
    items = []  # (set名, 質問文, 回答文リスト, ラベルリスト)
    sf = full["sf07"]["texts"]
    items.append(("sf07", qmap["sf07"], sf,
                  ["yes" if t.strip().lower().startswith("yes") else "double"
                   for t in sf]))
    po = full["po06"]["texts"]
    items.append(("po06", qmap["po06"], po, ["notdem"] * len(po)))
    sa = cpx["samples"]["complex_sales"]
    items.append(("sales", cpx["tasks"]["complex_sales"], sa,
                  ["ts" if any(k in t.lower() for k in
                               ("time se", "time-se", "trend", "sales tr"))
                   else "cust" for t in sa]))

    tok = AutoTokenizer.from_pretrained(args.model)
    kw = {"device_map": "auto"}
    if args.load_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kw["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()

    # 全層・全文平均の hidden state を回答ごとに取得
    per_set = {}
    n_layers = None
    for name, qtext, answers, labels in items:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": qtext}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        plen = tok(prompt, return_tensors="pt")["input_ids"].shape[1]
        vecs = []
        for ans in answers:
            ids = tok(prompt + ans, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**ids, output_hidden_states=True)
            layer_vecs = [h[0, plen:].float().mean(0).cpu().numpy()
                          for h in out.hidden_states]
            vecs.append(np.stack(layer_vecs))     # (層数, 次元)
        per_set[name] = (np.stack(vecs), labels)  # (回答数, 層数, 次元)
        n_layers = per_set[name][0].shape[1]
        print(f"[{name}] {len(answers)} answers x {n_layers} layers done")

    # 層ごとの指標
    metrics = []
    for li in range(n_layers):
        m = {"layer": li}
        # sf07: 立場間距離 / 言い回し内ブレ
        V, lb = per_set["sf07"]
        yes = V[[i for i, x in enumerate(lb) if x == "yes"], li]
        dbl = V[[i for i, x in enumerate(lb) if x == "double"], li]
        m["sf07_between"] = cosd(yes.mean(0), dbl.mean(0))
        m["sf07_within"] = (spread(yes) + spread(dbl)) / 2
        m["sf07_contrast"] = m["sf07_between"] / (m["sf07_within"] + 1e-9)
        # sales: 13対1
        V, lb = per_set["sales"]
        ts = V[[i for i, x in enumerate(lb) if x == "ts"], li]
        cu = V[[i for i, x in enumerate(lb) if x == "cust"], li]
        m["sales_between"] = cosd(ts.mean(0), cu.mean(0))
        m["sales_within"] = spread(ts)
        m["sales_contrast"] = m["sales_between"] / (m["sales_within"] + 1e-9)
        # po06: 同一立場の言い回しノイズ（sf07の立場間距離で相対化）
        V, _ = per_set["po06"]
        m["po06_within"] = spread(V[:, li])
        m["po06_rel_noise"] = m["po06_within"] / (m["sf07_between"] + 1e-9)
        metrics.append(m)

    best = max(metrics[1:], key=lambda m: m["sf07_contrast"])  # 0=埋め込み層は除外
    print(f"\n最適層(sf07立場分離度): L{best['layer']}/{n_layers-1} "
          f"contrast={best['sf07_contrast']:.1f} "
          f"(最終層={metrics[-1]['sf07_contrast']:.1f})")
    json.dump({"model": args.model, "n_layers": n_layers - 1,
               "metrics": metrics,
               "best_layer_sf07": best["layer"]},
              open(os.path.join(res, "layer_sweep.json"), "w"),
              ensure_ascii=False, indent=1)

    # fig08 再現用に sf07 の代表2層のベクトルを保存
    V, lb = per_set["sf07"]
    np.savez(os.path.join(res, "layer_sweep_sf07.npz"),
             best=V[:, best["layer"]], final=V[:, -1],
             labels=np.array(lb), best_layer=best["layer"])
    print(f"saved: {res}/layer_sweep.json, layer_sweep_sf07.npz")


if __name__ == "__main__":
    main()
