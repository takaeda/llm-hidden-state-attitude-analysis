#!/usr/bin/env python3
"""複雑なオープンエンド課題では木展開が破綻し、試行+hidden空間が必要になることを示す。

(i) 木展開の被覆率 vs 深さ:
    単純質問(st01)と複雑課題(分析方針の提案)で、サンプリングと同一の
    temperature/top-p 変換を適用した木を深さ8まで展開し、
    「展開できた確率質量(被覆率)」と「フロンティアの枝数」の推移を記録。
    forward回数に予算を設け、予算切れも現実の制約として記録する。

(ii) 複雑課題の意味構造:
    同じ課題に n回 フル回答を生成し、各回答の最終層hidden state(生成
    トークン平均)を保存。後段で2D化し意味クラスタを可視化する。

usage:
  python 12_complex_task.py --load-4bit
"""
import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TASKS = [
    ("simple_st01", "Does the Earth revolve around the Sun?"),
    ("complex_sales",
     "Our company's monthly sales have been declining for six months. "
     "Propose the single most important analysis we should run first, "
     "and briefly explain why."),
    ("complex_cause",
     "Our bakery's monthly revenue suddenly dropped 30% over the last "
     "three months. What is the most likely cause? Give your single best "
     "hypothesis and briefly explain."),
]


def sampling_dist(logits, temperature, top_p):
    probs = torch.softmax(logits / temperature, dim=-1)
    sp, si = torch.sort(probs, descending=True)
    cum = torch.cumsum(sp, dim=-1)
    keep = cum - sp < top_p
    sp = sp * keep
    sp = sp / sp.sum()
    out = torch.zeros_like(probs)
    out[si] = sp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--top-m", type=int, default=5)
    ap.add_argument("--prune", type=float, default=0.001)
    ap.add_argument("--budget", type=int, default=600,
                    help="課題あたりの forward 回数上限")
    ap.add_argument("--n-samples", type=int, default=14)
    ap.add_argument("--max-new-tokens", type=int, default=90)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = os.path.join(base, "results", args.model.split("/")[-1])
    os.makedirs(outdir, exist_ok=True)

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

    def make_inputs(text):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        return tok(prompt, return_tensors="pt").to(model.device)

    # ---- (i) 木展開の被覆率 vs 深さ ----
    tree_curves = {}
    for name, text in TASKS:
        inp = make_inputs(text)
        base_ids = inp["input_ids"][0]
        frontier = [([], 1.0)]
        n_forward = 0
        curve = []
        exhausted = False
        for d in range(1, args.depth + 1):
            frontier.sort(key=lambda kv: -kv[1])  # 確率の大きい枝から展開
            nxt = []
            for prefix, p in frontier:
                if n_forward >= args.budget:
                    exhausted = True
                    break
                ids = torch.cat([base_ids,
                                 torch.tensor(prefix, dtype=base_ids.dtype,
                                              device=base_ids.device)])
                with torch.no_grad():
                    out = model(ids.unsqueeze(0))
                n_forward += 1
                dist = sampling_dist(out.logits[0, -1].float(),
                                     args.temperature, args.top_p)
                top = torch.topk(dist, args.top_m)
                for t, tp in zip(top.indices.tolist(), top.values.tolist()):
                    cp = p * tp
                    if cp >= args.prune:
                        nxt.append((prefix + [t], cp))
            frontier = nxt
            cov = float(sum(p for _, p in frontier))
            curve.append({"depth": d, "coverage": round(cov, 4),
                          "n_branches": len(frontier),
                          "n_forward_total": n_forward,
                          "budget_exhausted": exhausted})
            print(f"[{name}] depth={d} coverage={cov:.3f} "
                  f"branches={len(frontier)} forwards={n_forward}"
                  f"{' (budget切れ)' if exhausted else ''}")
            if exhausted or not frontier:
                break
        tree_curves[name] = {"task": text, "curve": curve,
                             "params": {"top_m": args.top_m,
                                        "prune": args.prune,
                                        "budget": args.budget}}

    # ---- (ii) 複雑課題のサンプル群と hidden state ----
    all_samples, all_vecs = {}, {}
    for name, text in TASKS[1:]:
        inp = make_inputs(text)
        plen = inp["input_ids"].shape[1]
        with torch.no_grad():
            gens = model.generate(**inp, do_sample=True,
                                  temperature=args.temperature, top_p=args.top_p,
                                  max_new_tokens=args.max_new_tokens,
                                  num_return_sequences=args.n_samples,
                                  pad_token_id=eos)
        texts, vecs = [], []
        for gi in range(args.n_samples):
            gen_ids = gens[gi][plen:]
            keep = (gen_ids != eos).nonzero()
            L = int(keep[-1]) + 1 if len(keep) else 1
            seq = gens[gi][: plen + L].unsqueeze(0)
            with torch.no_grad():
                out = model(seq, output_hidden_states=True)
            hs = out.hidden_states[-1][0, plen:].float().cpu().numpy()
            vecs.append(hs.mean(0))
            texts.append(tok.decode(gen_ids[:L], skip_special_tokens=True))
            print(f"[{name} sample {gi}] {texts[-1][:70]!r}")
        all_samples[name] = texts
        all_vecs[name] = np.vstack(vecs)

    np.savez(os.path.join(outdir, "complex_task_vectors.npz"), **all_vecs)
    json.dump({"tree_curves": tree_curves,
               "tasks": dict(TASKS),
               "samples": all_samples},
              open(os.path.join(outdir, "complex_task.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"saved: {outdir}/complex_task.json, complex_task_vectors.npz")


if __name__ == "__main__":
    main()
