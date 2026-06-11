#!/usr/bin/env python3
"""Local LLM の「出力一貫性」品質スコアを、モデル間比較可能な形で算出する。

精度ではなく一貫性（再現性）の評価。出力文の意味は読まない。

手順（提案1＋提案2）:
 1. アンカー対立ペア K 組で、各モデルの「意味的距離の単位」を作る:
      D_ref = median_k  dist(centroid(P_k), centroid(P'_k))   ← max でなく中央値（HBDIの脆さ修正）
 2. 安全弁 d' = D_ref / (アンカー内の試行ばらつき平均)
      d' が十分大きいときだけ、このモデルの規格化を信用する
 3. 各評価質問 q について:
      規格化inconsistency(q) = trial_spread(q) / D_ref
      trial_spread = n試行の first-3-token hidden state の平均ペアワイズ・コサイン距離

すべて「出だし3トークンの最終層hidden state」空間で測る。
"""
import argparse
import json
import os

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def cosd(a, b):
    a = a/(np.linalg.norm(a)+1e-9)
    b = b/(np.linalg.norm(b)+1e-9)
    return 1.0-float(a@b)


def mean_pairwise(vs):
    if len(vs) < 2:
        return 0.0
    vn = vs/(np.linalg.norm(vs, axis=1, keepdims=True)+1e-9)
    S = vn@vn.T
    return float(1-S[np.triu_indices(len(vs), 1)].mean())


def cloud(model, tok, text, n, no_think, eos):
    """質問 text に n回サンプリング生成し、各試行の first-3-token 平均hidden stateを返す"""
    msgs = [{"role": "user", "content": text}]
    try:
        prompt = tok.apply_chat_template(msgs, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if no_think:
        prompt = (prompt.rstrip()+"\n\n</think>\n\n" if prompt.rstrip().endswith("<think>")
                  else prompt+"<think>\n\n</think>\n\n")
    inp = tok(prompt, return_tensors="pt").to(model.device)
    plen = inp["input_ids"].shape[1]
    with torch.no_grad():
        gens = model.generate(**inp, do_sample=True, temperature=0.8, top_p=0.95,
            max_new_tokens=6, num_return_sequences=n, pad_token_id=eos)
    vs = []
    for gi in range(n):
        seq = gens[gi][:plen+3].unsqueeze(0)
        with torch.no_grad():
            out = model(seq, output_hidden_states=True)
        hs = out.hidden_states[-1][0, plen:plen+3].float().cpu().numpy()
        vs.append(hs.mean(0))
    return np.vstack(vs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    anchors = yaml.safe_load(open(os.path.join(base, "anchors.yaml")))["pairs"]
    qall = yaml.safe_load(open(os.path.join(base, "questions_v1.yaml")))
    items = [(q["id"], c, q["text"]) for c in qall for q in qall[c]]
    tag = args.model.split("/")[-1]
    outdir = os.path.join(base, "results", tag)
    os.makedirs(outdir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    kw = {"device_map": "auto"}
    if args.load_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kw["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()
    eos = tok.eos_token_id

    # --- アンカーで物差しを作る ---
    d_between, within_anchor = [], []
    for pi, pair in enumerate(anchors):
        cT = cloud(model, tok, pair["pos"], args.n, args.no_think, eos)
        cF = cloud(model, tok, pair["neg"], args.n, args.no_think, eos)
        d = cosd(cT.mean(0), cF.mean(0))
        w = (mean_pairwise(cT)+mean_pairwise(cF))/2
        d_between.append(d)
        within_anchor.append(w)
        print(f"  anchor[{pi+1:2d}] between={d:.4f} within={w:.4f}  ({pair['pos'][:30]})")
    D_ref = float(np.median(d_between))
    noise = float(np.mean(within_anchor))
    d_prime = D_ref/(noise+1e-9)

    # --- 評価質問の規格化inconsistency ---
    per_q = {}
    for ki, (qid, cat, qtext) in enumerate(items):
        c = cloud(model, tok, qtext, args.n, args.no_think, eos)
        sp = mean_pairwise(c)
        per_q[qid] = {"category": cat, "trial_spread": round(sp, 5),
                      "norm_inconsistency": round(sp/(D_ref+1e-9), 4)}
        print(f"[{ki+1:2d}/36] {qid} {cat:13s} spread={sp:.4f} norm={sp/(D_ref+1e-9):.3f}")

    cats = ["science_true", "science_false", "contested", "political"]
    by_cat = {c: round(float(np.mean([v["norm_inconsistency"] for v in per_q.values()
              if v["category"] == c])), 4) for c in cats}
    overall = round(float(np.mean([v["norm_inconsistency"] for v in per_q.values()])), 4)

    summary = {
        "model": args.model, "n": args.n,
        "D_ref_anchor_unit": round(D_ref, 5),
        "anchor_noise_floor": round(noise, 5),
        "d_prime_validity": round(d_prime, 3),
        "valid": d_prime > 3.0,   # 物差しが信用できるか
        "norm_inconsistency_overall": overall,
        "norm_inconsistency_by_category": by_cat,
        "anchor_between_distances": [round(x, 4) for x in d_between],
        "per_question": per_q,
    }
    json.dump(summary, open(os.path.join(outdir, "consistency.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\n=== {tag} ===")
    print(f"D_ref(意味的距離の単位)={D_ref:.4f}  noise={noise:.4f}  "
          f"d'={d_prime:.2f}  {'[物差し有効]' if d_prime>3 else '[物差し疑わしい]'}")
    print(f"規格化inconsistency 全体={overall:.3f}  カテゴリ別={by_cat}")
    print(f"  （小さいほど一貫＝定型業務に載せやすい。正誤とは無関係）")


if __name__ == "__main__":
    main()
