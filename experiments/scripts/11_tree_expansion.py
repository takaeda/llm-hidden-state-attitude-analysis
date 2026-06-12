#!/usr/bin/env python3
"""浅い木展開 vs サンプリングの検証。

出だし3トークンの分布を:
 (A) 木展開で厳密に計算（各ノードで全語彙softmax → top-m枝を展開、
     サンプリングと同一の temperature / top-p 変換を適用。
     列挙した枝の確率は厳密、未展開分は残差として既知）
 (B) 同条件のサンプリング n回 で頻度推定（モンテカルロ）
し、(A)と(B)が二項誤差の範囲で一致するかを確認する。

usage:
  python 11_tree_expansion.py --model Qwen/Qwen3-4B --load-4bit \
      --qids st01,sf07,po06 --depth 3 --top-m 5 --n-samples 40
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def sampling_dist(logits, temperature, top_p):
    """HF generate の do_sample と同じ変換を施した次トークン分布を返す"""
    probs = torch.softmax(logits / temperature, dim=-1)
    sp, si = torch.sort(probs, descending=True)
    cum = torch.cumsum(sp, dim=-1)
    keep = cum - sp < top_p  # top-p 核: 累積がtop_pを超える直前まで残す
    sp = sp * keep
    sp = sp / sp.sum()
    out = torch.zeros_like(probs)
    out[si] = sp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--qids", default="st01,sf07,po06")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--top-m", type=int, default=5)
    ap.add_argument("--prune", type=float, default=0.002,
                    help="この累積確率未満の枝は展開しない（残差に計上）")
    ap.add_argument("--n-samples", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    qall = yaml.safe_load(open(os.path.join(base, "questions_v1.yaml")))
    qmap = {q["id"]: q["text"] for c in qall for q in qall[c]}

    tok = AutoTokenizer.from_pretrained(args.model)
    kw = {"device_map": "auto"}
    if args.load_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kw["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()
    eos = tok.eos_token_id

    report = {}
    for qid in args.qids.split(","):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": qmap[qid]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inp = tok(prompt, return_tensors="pt").to(model.device)
        base_ids = inp["input_ids"][0]

        # --- (A) 木展開（厳密） ---
        frontier = [([], 1.0)]   # (生成済みトークン列, 厳密確率)
        pruned_mass = 0.0
        n_forward = 0
        for d in range(args.depth):
            nxt = []
            for prefix, p in frontier:
                ids = torch.cat([base_ids,
                                 torch.tensor(prefix, dtype=base_ids.dtype,
                                              device=base_ids.device)])
                with torch.no_grad():
                    out = model(ids.unsqueeze(0))
                n_forward += 1
                dist = sampling_dist(out.logits[0, -1].float(),
                                     args.temperature, args.top_p)
                top = torch.topk(dist, args.top_m)
                covered = 0.0
                for t, tp in zip(top.indices.tolist(), top.values.tolist()):
                    if tp <= 0:
                        continue
                    covered += tp
                    child_p = p * tp
                    if child_p < args.prune:
                        pruned_mass += child_p
                    else:
                        nxt.append((prefix + [t], child_p))
                pruned_mass += p * max(0.0, 1.0 - covered)  # top-m外（核内残り）
            frontier = nxt
        leaves = {tok.decode(pre): p for pre, p in frontier}
        coverage = sum(leaves.values())

        # --- (B) サンプリング（同条件） ---
        with torch.no_grad():
            gens = model.generate(
                **inp, do_sample=True, temperature=args.temperature,
                top_p=args.top_p, max_new_tokens=args.depth,
                num_return_sequences=args.n_samples, pad_token_id=eos)
        plen = inp["input_ids"].shape[1]
        cnt = Counter(tok.decode(gens[i][plen:plen + args.depth])
                      for i in range(args.n_samples))

        # --- 突き合わせ ---
        n = args.n_samples
        rows = []
        for text, p in sorted(leaves.items(), key=lambda kv: -kv[1]):
            f = cnt.get(text, 0) / n
            se = float(np.sqrt(max(p * (1 - p), 1e-9) / n))  # 二項標準誤差
            rows.append({"prefix": text, "exact_p": round(p, 4),
                         "sampled_f": round(f, 4),
                         "binom_se": round(se, 4),
                         "within_2se": bool(abs(f - p) <= 2 * se)})
        # サンプルにあるが木に無いもの（残差の中身の実例）
        extra = {t: c / n for t, c in cnt.items() if t not in leaves}

        report[qid] = {
            "question": qmap[qid],
            "tree": {"depth": args.depth, "top_m": args.top_m,
                     "n_forward": n_forward,
                     "coverage": round(coverage, 4),
                     "residual_known": round(1 - coverage, 4)},
            "comparison": rows,
            "sampled_only_prefixes": {k: round(v, 3) for k, v in extra.items()},
        }
        print(f"\n=== {qid}: {qmap[qid]}")
        print(f"  木展開: forward {n_forward}回, 被覆率 {coverage:.1%} "
              f"(残差 {1-coverage:.1%} は量として既知)")
        print(f"  {'プレフィクス':30s} {'厳密p':>7s} {'標本f':>7s} {'2SE内':>5s}")
        for r in rows[:8]:
            print(f"  {r['prefix']!r:32s} {r['exact_p']:7.3f} "
                  f"{r['sampled_f']:7.3f} {'OK' if r['within_2se'] else 'NG':>5s}")
        if extra:
            print(f"  (標本のみ: {extra})")

    out = os.path.join(base, "results", args.model.split("/")[-1],
                       "tree_vs_sampling.json")
    json.dump(report, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
