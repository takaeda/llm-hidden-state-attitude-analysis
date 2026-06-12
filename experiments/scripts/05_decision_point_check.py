#!/usr/bin/env python3
"""決定点のhidden stateのゆらぎを分解する。

各質問にn回サンプリング生成し、各試行で2つの位置の最終層hidden stateを取る:
  (P) プロンプト末尾位置（1語目を予測する状態）
  (T) 1語目トークンの位置（2語目を予測する状態）
そして:
  - (P)が試行間でゆらぐか（理論上は決定的→ゆらがないはず）
  - (T)のゆらぎが「1語目の文字(yes/no/other)」だけで決まるか、
    同じ1語目の中にも構造（態度）が残るか
を、群間/群内の散らばりに分解して確認する。
"""
import argparse
import json
import os
import re

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

YES = re.compile(r"^(yes|certainly|definitely|absolutely|indeed|sure|correct|true)", re.I)
NO = re.compile(r"^(no|nope|incorrect|false)", re.I)


def tok_cat(s):
    t = s.strip().lower()
    if YES.match(t):
        return "yes"
    if NO.match(t):
        return "no"
    return "other"


def cosd(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return 1.0 - float(a @ b)


def spread(vs):
    if len(vs) < 2:
        return 0.0
    vn = vs / (np.linalg.norm(vs, axis=1, keepdims=True) + 1e-9)
    S = vn @ vn.T
    iu = np.triu_indices(len(vs), 1)
    return float(1 - S[iu].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--qids", default="st01,po01,sf07,po06")
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    qall = yaml.safe_load(open(os.path.join(base, "questions_v1.yaml")))
    qmap = {q["id"]: q["text"] for c in qall for q in qall[c]}

    tok = AutoTokenizer.from_pretrained(args.model)
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=qc, device_map="auto").eval()
    eos = tok.eos_token_id

    report = {}
    for qid in args.qids.split(","):
        msgs = [{"role": "user", "content": qmap[qid]}]
        try:
            prompt = tok.apply_chat_template(msgs, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(msgs, tokenize=False,
                add_generation_prompt=True)
        inp = tok(prompt, return_tensors="pt").to(model.device)
        plen = inp["input_ids"].shape[1]

        with torch.no_grad():
            gens = model.generate(**inp, do_sample=True, temperature=args.temperature,
                top_p=0.95, max_new_tokens=4, num_return_sequences=args.n,
                pad_token_id=eos)

        P_states, T_states, first_toks = [], [], []
        for gi in range(gens.shape[0]):
            seq = gens[gi][: plen + 2].unsqueeze(0)
            with torch.no_grad():
                out = model(seq, output_hidden_states=True)
            hs = out.hidden_states[-1][0].float().cpu().numpy()
            P_states.append(hs[plen - 1])          # プロンプト末尾位置
            T_states.append(hs[plen])              # 1語目トークン位置
            first_toks.append(tok.decode([int(gens[gi][plen])]))
        P_states = np.vstack(P_states)
        T_states = np.vstack(T_states)
        cats = [tok_cat(t) for t in first_toks]

        # (P)のゆらぎ
        p_spread = spread(P_states)
        # (T)を1語目カテゴリで群分け
        from collections import Counter
        cnt = Counter(cats)
        within = []
        centroids = {}
        for c in cnt:
            idx = [i for i, x in enumerate(cats) if x == c]
            within.append(spread(T_states[idx]))
            centroids[c] = T_states[idx].mean(0)
        within_mean = float(np.mean(within)) if within else 0.0
        if len(centroids) >= 2:
            cs = list(centroids.values())
            between = float(np.mean([cosd(cs[i], cs[j])
                for i in range(len(cs)) for j in range(i + 1, len(cs))]))
        else:
            between = 0.0
        T_total = spread(T_states)

        report[qid] = {
            "question": qmap[qid],
            "first_token_distribution": dict(cnt),
            "P_promptend_spread": round(p_spread, 6),
            "T_firsttoken_total_spread": round(T_total, 5),
            "T_within_token_spread": round(within_mean, 6),
            "T_between_token_spread": round(between, 5),
        }
        print(f"\n=== {qid}: {qmap[qid]}")
        print(f"  1語目の分布: {dict(cnt)}")
        print(f"  (P)プロンプト末尾のゆらぎ      = {p_spread:.6f}  ← 理論上ほぼ0のはず")
        print(f"  (T)1語目位置 全体のゆらぎ       = {T_total:.5f}")
        print(f"      ├ 同じ1語目の中(群内)       = {within_mean:.6f}")
        print(f"      └ 1語目どうしの差(群間)     = {between:.5f}")

    out = os.path.join(base, "results", args.model.split("/")[-1],
                       "decision_point_check.json")
    json.dump(report, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
