#!/usr/bin/env python3
"""出だし何語まで見るかで hidden state の広がりがどう変わるかを測る。

各質問に n 回サンプリング生成し、各試行で「最初の k トークンの最終層hidden state
を平均」したベクトルを作る。k = 1,3,5,8,全体 それぞれで試行間の広がりを測る。

狙い:
 - k=1 はトークンの文字で決まる（前実験で確認）→ 広がりは離散的
 - k=3 は出だしの分岐を捉える（教材の方法論の核）
 - k=全体 は作文の多様性に汚染される（po06の交絡）
どの窓が確信度（行動の迷い）に最もきれいに対応するかを見る。
"""
import csv
import json
import os
import re

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

YES_START = re.compile(r"^(yes|certainly|definitely|absolutely|indeed|of course|correct)\b", re.I)
NO_START = re.compile(r"^(no|nope|incorrect)\b", re.I)
NO_CONTENT = re.compile(r"\b(is not|are not|was not|were not|does not|do not|did not)\b", re.I)
YES_CONTENT = re.compile(r"\b(the answer is yes|it is true that)\b", re.I)


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
    h = -sum((v/tot)*np.log(v/tot) for v in c.values() if v)
    return float(h/np.log(3))


def spread(vs):
    if len(vs) < 2:
        return 0.0
    vn = vs/(np.linalg.norm(vs, axis=1, keepdims=True)+1e-9)
    S = vn@vn.T
    return float(1-S[np.triu_indices(len(vs), 1)].mean())


SUBSET = ["st01", "st02", "st08", "sf01", "sf02", "sf07",
          "ct01", "ct04", "ct06", "po01", "po02", "po06"]
WINDOWS = [1, 3, 5, 8]


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    qall = yaml.safe_load(open(os.path.join(base, "questions_v1.yaml")))
    qmap = {q["id"]: q["text"] for c in qall for q in qall[c]}
    catmap = {q["id"]: c for c in qall for q in qall[c]}

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B", quantization_config=qc, device_map="auto").eval()
    eos = tok.eos_token_id
    N = 20

    per_q = {}
    csv3 = []   # k=3 ベクトル（toorPIA用）
    for ki, qid in enumerate(SUBSET):
        msgs = [{"role": "user", "content": qmap[qid]}]
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        inp = tok(prompt, return_tensors="pt").to(model.device)
        plen = inp["input_ids"].shape[1]
        with torch.no_grad():
            gens = model.generate(**inp, do_sample=True, temperature=0.8, top_p=0.95,
                                  max_new_tokens=40, num_return_sequences=N, pad_token_id=eos)
        win_vecs = {k: [] for k in WINDOWS}
        full_vecs = []
        counts = {"YES": 0, "NO": 0, "HEDGE": 0}
        for gi in range(N):
            gen_ids = gens[gi][plen:]
            keep = (gen_ids != eos).nonzero()
            L = int(keep[-1])+1 if len(keep) else 1
            seq = gens[gi][:plen+L].unsqueeze(0)
            with torch.no_grad():
                out = model(seq, output_hidden_states=True)
            hs = out.hidden_states[-1][0, plen:].float().cpu().numpy()  # (L,dim)
            for k in WINDOWS:
                win_vecs[k].append(hs[:min(k, len(hs))].mean(0))
            full_vecs.append(hs.mean(0))
            counts[classify(tok.decode(gen_ids[:L], skip_special_tokens=True))] += 1
            csv3.append((qid, catmap[qid], gi, hs[:min(3, len(hs))].mean(0)))
        rec = {"question": qmap[qid], "category": catmap[qid],
               "behavior_entropy": round(entropy(counts), 4), "counts": counts,
               "spread": {f"k{k}": round(spread(np.vstack(win_vecs[k])), 5) for k in WINDOWS}}
        rec["spread"]["full"] = round(spread(np.vstack(full_vecs)), 5)
        per_q[qid] = rec
        s = rec["spread"]
        print(f"[{ki+1:2d}/12] {qid} {catmap[qid]:13s} behH={rec['behavior_entropy']:.2f} "
              f"k1={s['k1']:.4f} k3={s['k3']:.4f} k5={s['k5']:.4f} full={s['full']:.4f}")

    # 相関: 各窓でのspread vs 行動エントロピー
    qs = list(per_q.values())
    cors = {}
    for k in [f"k{w}" for w in WINDOWS]+["full"]:
        xs = [q["spread"][k] for q in qs]
        ys = [q["behavior_entropy"] for q in qs]
        cors[k] = round(float(np.corrcoef(xs, ys)[0, 1]), 3)
    summary = {"correlation_spread_vs_behaviorH": cors, "per_question": per_q}
    out = os.path.join(base, "results", "Qwen3-4B")
    json.dump(summary, open(os.path.join(out, "window_spread.json"), "w"),
              ensure_ascii=False, indent=1)

    dim = csv3[0][3].shape[0]
    with open(os.path.join(out, "first3_vectors.csv"), "w") as f:
        f.write("qid,category,trial,"+",".join(f"d{i}" for i in range(dim))+"\n")
        for qid, cat, gi, v in csv3:
            f.write(f"{qid},{cat},{gi},"+",".join(f"{x:.5f}" for x in v)+"\n")

    print(f"\n相関(spread vs 行動の迷い): {cors}")
    print("po06の窓別spread:", per_q["po06"]["spread"], "← full↓なら交絡解消")
    print("st01の窓別spread:", per_q["st01"]["spread"], "← 確信質問")
    print(f"saved: window_spread.json, first3_vectors.csv")


if __name__ == "__main__":
    main()
