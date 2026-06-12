#!/usr/bin/env python3
"""プーリング方式の比較: 「立場の分離」と「言い回しノイズ」を最もよく
切り分けるのは、どの位置・どの層の hidden state か。

比較する取り方（層 × 位置）:
  位置: first3(出だし3語平均) / full(全文平均) / last(文末トークン)
  層:   final(最終層) / mid(中間層 L20)

素材:
  sf07 (DNA三重らせん): 立場が割れる(Yes×2/二重らせん×10)。
        二重らせん内には書き出し2系統("DNA typically"/"DNA (deoxy")の
        言い回し分裂があり、within に言い回しノイズが含まれる
  po06 (中国は民主主義): 立場は同一・言い回しのみ多様

指標:
  between      = sf07 立場間の重心コサイン距離
  within_sf07  = sf07 同一立場内の平均散らばり
  contrast     = between / within_sf07           (高いほど良い)
  po06_within  = po06 の散らばり（純粋な言い回しノイズ）
  rel_noise    = po06_within / between           (低いほど良い)

usage: python 14_pooling_compare.py --load-4bit
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
    ap.add_argument("--mid-layer", type=int, default=20)
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res = os.path.join(base, "results", args.model.split("/")[-1])
    qall = yaml.safe_load(open(os.path.join(base, "questions_v1.yaml")))
    qmap = {q["id"]: q["text"] for c in qall for q in qall[c]}
    full = json.load(open(os.path.join(res, "full_texts.json")))

    tok = AutoTokenizer.from_pretrained(args.model)
    kw = {"device_map": "auto"}
    if args.load_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kw["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()

    LAYERS = {"mid": args.mid_layer, "final": None}  # None=最後

    def pooled_vectors(qid):
        """各回答について {(pos,layer): vec} を返す"""
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": qmap[qid]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        plen = tok(prompt, return_tensors="pt")["input_ids"].shape[1]
        out_list = []
        for ans in full[qid]["texts"]:
            ids = tok(prompt + ans, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**ids, output_hidden_states=True)
            d = {}
            for lname, li in LAYERS.items():
                h = out.hidden_states[li if li is not None else -1][0]
                gen = h[plen:].float().cpu().numpy()
                d[("first3", lname)] = gen[:3].mean(0)
                d[("full", lname)] = gen.mean(0)
                d[("last", lname)] = gen[-1]
            out_list.append(d)
        return out_list

    sf = pooled_vectors("sf07")
    po = pooled_vectors("po06")
    yes_idx = [i for i, t in enumerate(full["sf07"]["texts"])
               if t.strip().lower().startswith("yes")]
    dbl_idx = [i for i in range(len(sf)) if i not in yes_idx]

    rows = []
    for pos in ("first3", "full", "last"):
        for lname in ("final", "mid"):
            key = (pos, lname)
            Y = np.stack([sf[i][key] for i in yes_idx])
            D = np.stack([sf[i][key] for i in dbl_idx])
            P = np.stack([p[key] for p in po])
            between = cosd(Y.mean(0), D.mean(0))
            within = (spread(Y) + spread(D)) / 2
            po_w = spread(P)
            rows.append({
                "pooling": pos, "layer": lname,
                "sf07_between": round(between, 4),
                "sf07_within": round(within, 4),
                "contrast": round(between / (within + 1e-9), 2),
                "po06_within": round(po_w, 4),
                "rel_noise": round(po_w / (between + 1e-9), 3),
            })

    print(f"\n{'位置':8s} {'層':6s} {'立場間':>7s} {'立場内':>7s} "
          f"{'分離度↑':>7s} {'po06ノイズ':>9s} {'相対ノイズ↓':>9s}")
    for r in rows:
        print(f"{r['pooling']:8s} {r['layer']:6s} {r['sf07_between']:7.4f} "
              f"{r['sf07_within']:7.4f} {r['contrast']:7.2f} "
              f"{r['po06_within']:9.4f} {r['rel_noise']:9.3f}")

    json.dump({"model": args.model, "mid_layer": args.mid_layer,
               "rows": rows},
              open(os.path.join(res, "pooling_compare.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\nsaved: {res}/pooling_compare.json")


if __name__ == "__main__":
    main()
