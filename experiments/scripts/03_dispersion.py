#!/usr/bin/env python3
"""hidden state分布の「広がり」が確信度を表すかを検証する。

各質問にn回サンプリング生成し、各試行の回答トークンの最終層hidden stateを
平均プールして1ベクトルにする → n本の点群を得る。
- 広がり spread = 平均ペアワイズ・コサイン距離
- 行動エントロピー = n回の回答内容(YES/NO/HEDGE)の割れ方
- 対立ペアは並べ替え検定で「中心間距離が群内ノイズに対し有意か」を判定

教材方式（生のhidden stateベクトル）を用いる。logit lensは使わない。

usage:
  python 03_dispersion.py --model Qwen/Qwen3-4B --load-4bit --n-trials 20
"""

import argparse
import json
import os
import re

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

# 行動分類（02_agreementと同一ルール）
YES_START = re.compile(
    r"^(yes|certainly|definitely|absolutely|indeed|of course|correct)\b", re.I)
NO_START = re.compile(r"^(no|nope|incorrect)\b", re.I)
YES_CONTENT = re.compile(r"\b(the answer is yes|answer:\s*yes|it is true that)\b", re.I)
NO_CONTENT = re.compile(
    r"\b(the answer is no|answer:\s*no|is not|are not|was not|were not"
    r"|does not|do not|did not)\b", re.I)


def classify(text):
    t = re.sub(r'[*_#>"\'`]', "", text.strip()).strip()
    if YES_START.match(t):
        return "YES"
    if NO_START.match(t):
        return "NO"
    head = t[:200]
    if YES_CONTENT.search(head):
        return "YES"
    if NO_CONTENT.search(head):
        return "NO"
    return "HEDGE"


def entropy(counts):
    tot = sum(counts.values())
    if tot == 0:
        return 0.0
    h = 0.0
    for v in counts.values():
        if v > 0:
            p = v / tot
            h -= p * np.log(p)
    return float(h / np.log(3))  # 0..1 に正規化(3クラス)


def cos_dist(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return 1.0 - float(np.dot(a, b))


def mean_pairwise_cos(vecs):
    n = len(vecs)
    if n < 2:
        return 0.0
    vn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    sims = vn @ vn.T
    iu = np.triu_indices(n, k=1)
    return float(1.0 - sims[iu].mean())


def within_spread(vecs):
    c = vecs.mean(0)
    return float(np.mean([cos_dist(v, c) for v in vecs]))


def perm_test(A, B, n_perm=2000, seed=0):
    rng = np.random.default_rng(seed)
    obs = cos_dist(A.mean(0), B.mean(0))
    pooled = np.vstack([A, B])
    n = len(A)
    cnt = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(pooled))
        if cos_dist(pooled[idx[:n]].mean(0), pooled[idx[n:]].mean(0)) >= obs:
            cnt += 1
    null = []
    rng2 = np.random.default_rng(seed + 1)
    for _ in range(500):
        idx = rng2.permutation(len(pooled))
        null.append(cos_dist(pooled[idx[:n]].mean(0), pooled[idx[n:]].mean(0)))
    win = (within_spread(A) + within_spread(B)) / 2
    return {
        "obs_dist": round(obs, 5),
        "p_value": (cnt + 1) / (n_perm + 1),
        "null_mean_dist": round(float(np.mean(null)), 5),
        "within_spread": round(win, 5),
        "effect_size": round(obs / (win + 1e-9), 3),
    }


SUBSET = ["st01", "st02", "st08", "sf01", "sf02", "sf07",
          "ct01", "ct04", "ct06", "po01", "po02", "po06"]
PAIRS = [("po01", "po02", "Taiwan independent vs part-of-China"),
         ("st01", "sf01", "Earth->Sun vs Sun->Earth")]


def build_inputs(tokenizer, q, device, no_think=False):
    msgs = [{"role": "user", "content": q}]
    try:
        p = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        p = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
    if no_think:
        p = (p.rstrip() + "\n\n</think>\n\n" if p.rstrip().endswith("<think>")
             else p + "<think>\n\n</think>\n\n")
    return tokenizer(p, return_tensors="pt").to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--subset", default=None,
                    help="カンマ区切りのqid。省略時は既定12問")
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tag = args.model.split("/")[-1]
    outdir = os.path.join(base, "results", tag)
    os.makedirs(outdir, exist_ok=True)

    qall = yaml.safe_load(open(os.path.join(base, "questions_v1.yaml")))
    qmap = {q["id"]: q["text"] for cat in qall for q in qall[cat]}
    catmap = {q["id"]: cat for cat in qall for q in qall[cat]}
    subset = args.subset.split(",") if args.subset else SUBSET

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=qc, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    eos = tokenizer.eos_token_id

    per_q = {}
    vec_store = {}
    csv_rows = []
    for k, qid in enumerate(subset):
        qtext = qmap[qid]
        inputs = build_inputs(tokenizer, qtext, model.device, args.no_think)
        plen = inputs["input_ids"].shape[1]
        with torch.no_grad():
            gens = model.generate(
                **inputs, do_sample=True, temperature=args.temperature,
                top_p=0.95, max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.n_trials, pad_token_id=eos)
        vecs = []
        counts = {"YES": 0, "NO": 0, "HEDGE": 0}
        for gi in range(gens.shape[0]):
            seq = gens[gi]
            gen_ids = seq[plen:]
            # 末尾padを除去
            keep = (gen_ids != eos).nonzero()
            last = int(keep[-1]) + 1 if len(keep) else 1
            full = seq[: plen + max(last, 1)].unsqueeze(0)
            with torch.no_grad():
                out = model(full, output_hidden_states=True)
            hs = out.hidden_states[-1][0]            # (seqlen, dim)
            gen_hs = hs[plen:].float().cpu().numpy()  # 生成位置のみ
            vec = gen_hs.mean(0)                      # 平均プール
            vecs.append(vec)
            ans = tokenizer.decode(gen_ids[:last], skip_special_tokens=True)
            counts[classify(ans)] += 1
            csv_rows.append((qid, catmap[qid], gi, vec))
        vecs = np.vstack(vecs)
        vec_store[qid] = vecs
        spread = mean_pairwise_cos(vecs)
        h = entropy(counts)
        per_q[qid] = {
            "question": qtext, "category": catmap[qid],
            "spread_mean_pairwise_cos": round(spread, 5),
            "within_spread": round(within_spread(vecs), 5),
            "behavior_counts": counts, "behavior_entropy": round(h, 4),
        }
        print(f"[{k+1:2d}/{len(subset)}] {qid} {catmap[qid]:13s} "
              f"spread={spread:.4f} beh_H={h:.3f} {counts}")

    # 対立ペア検定
    pair_res = {}
    for a, b, name in PAIRS:
        if a in vec_store and b in vec_store:
            pair_res[name] = {
                "qid_A": a, "qid_B": b,
                **perm_test(vec_store[a], vec_store[b]),
            }

    # H1相関
    qs = list(per_q.values())
    rho_spr_h = float(np.corrcoef(
        [q["spread_mean_pairwise_cos"] for q in qs],
        [q["behavior_entropy"] for q in qs])[0, 1])

    summary = {
        "model": args.model, "n_trials": args.n_trials,
        "spread_vs_behavior_entropy_pearson": round(rho_spr_h, 3),
        "spread_by_category": {
            c: round(float(np.mean(
                [v["spread_mean_pairwise_cos"] for v in per_q.values()
                 if v["category"] == c])), 4)
            for c in ["science_true", "science_false", "contested", "political"]},
        "per_question": per_q,
        "pair_tests": pair_res,
    }
    json.dump(summary, open(os.path.join(outdir, "dispersion.json"), "w"),
              ensure_ascii=False, indent=1)

    # toorPIA用CSV
    dim = csv_rows[0][3].shape[0]
    with open(os.path.join(outdir, "dispersion_vectors.csv"), "w") as f:
        f.write("qid,category,trial," + ",".join(f"d{i}" for i in range(dim)) + "\n")
        for qid, cat, gi, vec in csv_rows:
            f.write(f"{qid},{cat},{gi}," +
                    ",".join(f"{x:.5f}" for x in vec) + "\n")

    print(f"\n=== {tag} ===")
    print(f"spread by category: {summary['spread_by_category']}")
    print(f"Pearson(spread, behavior entropy) = {rho_spr_h:.3f}")
    for name, r in pair_res.items():
        print(f"PAIR {name}: dist={r['obs_dist']} p={r['p_value']:.4f} "
              f"effect={r['effect_size']} (null~{r['null_mean_dist']})")
    print(f"saved: {outdir}/dispersion.json, dispersion_vectors.csv")


if __name__ == "__main__":
    main()
