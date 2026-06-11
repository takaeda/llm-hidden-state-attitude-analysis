#!/usr/bin/env python3
"""1モデルの「出だし3語の広がり」プロファイルを作る。

全36問に n 回サンプリング生成し、各質問で:
 - 出だし3トークンの最終層hidden stateを平均 → 試行間の広がり(k3 spread)
   = そのモデルがどれだけ一貫した出だしで答え始めるか（立場へのコミットの強さ）
 - 回答内容の YES/NO/HEDGE 分類 → 行動の迷い・立場・(事実問題なら)正誤
を記録する。カテゴリ別に集計してモデルの個性を出す。

usage:
  python 07_model_profile.py --model Qwen/Qwen3-4B --load-4bit
  python 07_model_profile.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --no-think
"""
import argparse
import json
import os
import re

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

YES_START = re.compile(r"^(yes|certainly|definitely|absolutely|indeed|of course|correct)\b", re.I)
NO_START = re.compile(r"^(no|nope|incorrect)\b", re.I)
NO_CONTENT = re.compile(r"\b(is not|are not|was not|were not|does not|do not|did not|never)\b", re.I)
YES_CONTENT = re.compile(r"\b(the answer is yes|it is true that|that is correct)\b", re.I)

CORRECT = {"science_true": "YES", "science_false": "NO"}  # 事実問題の正解


def classify(t):
    t = re.sub(r'[*_#>"\'`]', "", t.strip()).strip()
    if YES_START.match(t):
        return "YES"
    if NO_START.match(t):
        return "NO"
    if YES_CONTENT.search(t[:200]):
        return "YES"
    if NO_CONTENT.search(t[:200]):
        return "NO"
    return "HEDGE"


def entropy(c):
    tot = sum(c.values())
    if not tot:
        return 0.0
    return float(-sum((v/tot)*np.log(v/tot) for v in c.values() if v)/np.log(3))


def spread(vs):
    if len(vs) < 2:
        return 0.0
    vn = vs/(np.linalg.norm(vs, axis=1, keepdims=True)+1e-9)
    S = vn@vn.T
    return float(1-S[np.triu_indices(len(vs), 1)].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=36)
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    qall = yaml.safe_load(open(os.path.join(base, "questions_v1.yaml")))
    items = [(q["id"], c, q["text"]) for c in qall for q in qall[c]]
    tag = args.model.split("/")[-1]
    outdir = os.path.join(base, "results", tag)
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

    per_q = {}
    for ki, (qid, cat, qtext) in enumerate(items):
        msgs = [{"role": "user", "content": qtext}]
        try:
            prompt = tok.apply_chat_template(msgs, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if args.no_think:
            prompt = (prompt.rstrip()+"\n\n</think>\n\n" if prompt.rstrip().endswith("<think>")
                      else prompt+"<think>\n\n</think>\n\n")
        inp = tok(prompt, return_tensors="pt").to(model.device)
        plen = inp["input_ids"].shape[1]
        with torch.no_grad():
            gens = model.generate(**inp, do_sample=True, temperature=0.8, top_p=0.95,
                max_new_tokens=args.max_new_tokens, num_return_sequences=args.n,
                pad_token_id=eos)
        v3 = []
        counts = {"YES": 0, "NO": 0, "HEDGE": 0}
        for gi in range(args.n):
            gen_ids = gens[gi][plen:]
            keep = (gen_ids != eos).nonzero()
            L = int(keep[-1])+1 if len(keep) else 1
            seq = gens[gi][:plen+min(L, 3)].unsqueeze(0)
            with torch.no_grad():
                out = model(seq, output_hidden_states=True)
            hs = out.hidden_states[-1][0, plen:].float().cpu().numpy()
            v3.append(hs[:min(3, len(hs))].mean(0))
            counts[classify(tok.decode(gen_ids[:L], skip_special_tokens=True))] += 1
        k3 = spread(np.vstack(v3))
        acc = (counts.get(CORRECT[cat], 0)/args.n) if cat in CORRECT else None
        per_q[qid] = {"category": cat, "question": qtext, "k3_spread": round(k3, 5),
                      "counts": counts, "behavior_entropy": round(entropy(counts), 4),
                      "accuracy": acc}
        print(f"[{ki+1:2d}/36] {qid} {cat:13s} k3={k3:.4f} {counts} "
              f"acc={acc if acc is None else round(acc,2)}")

    cats = ["science_true", "science_false", "contested", "political"]
    by_cat = {c: round(float(np.mean([v["k3_spread"] for v in per_q.values()
              if v["category"] == c])), 5) for c in cats}
    sci = float(np.mean([by_cat["science_true"], by_cat["science_false"]]))
    pol = by_cat["political"]
    fact_acc = float(np.mean([v["accuracy"] for v in per_q.values()
                    if v["accuracy"] is not None]))
    pol_decisive = float(np.mean([1-(v["counts"]["HEDGE"]/sum(v["counts"].values()))
                          for v in per_q.values() if v["category"] == "political"]))
    summary = {
        "model": args.model, "n": args.n,
        "k3_spread_by_category": by_cat,
        "science_spread": round(sci, 5), "political_spread": round(pol, 5),
        "political_commitment_index": round(sci/(pol+1e-9), 3),  # >1: 政治でも科学並みに頑な
        "factual_accuracy": round(fact_acc, 3),
        "political_decisiveness": round(pol_decisive, 3),  # 政治でYes/No言い切る率
        "per_question": per_q,
    }
    json.dump(summary, open(os.path.join(outdir, "profile.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\n=== {tag} ===")
    print(f"k3 spread by category: {by_cat}")
    print(f"science={sci:.4f} political={pol:.4f} "
          f"commitment_index(sci/pol)={sci/(pol+1e-9):.2f} "
          f"(大きいほど政治に頑な)")
    print(f"factual_accuracy={fact_acc:.2f} political_decisiveness={pol_decisive:.2f}")


if __name__ == "__main__":
    main()
