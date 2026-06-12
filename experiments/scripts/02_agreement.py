#!/usr/bin/env python3
"""一致率検証: 生成前のLogit Lens読み値（Yes/No/非コミット）が、
実際の回答行動（temperature サンプリング n回の回答内容の分類）と
どの程度一致するかを測る。

- 読み値（lens）: プロンプト末尾・最終層の次トークン分布から
    P(yes系), P(no系), P(その他=最初の語でコミットしない) の argmax
- 行動（behavior）: n回生成した回答全文をルールで YES/NO/HEDGE に分類し、
    その多数決クラスと割合

usage:
  python 02_agreement.py --model Qwen/Qwen3-4B --load-4bit --n-samples 10
"""

import argparse
import json
import os
import re

import torch
import yaml
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

YES_WORDS = ["Yes", " Yes", "yes", " yes"]
NO_WORDS = ["No", " No", "no", " no"]

# --- 回答テキストの内容分類ルール ---
YES_START = re.compile(
    r"^(yes|certainly|definitely|absolutely|indeed|of course|correct)\b", re.I)
NO_START = re.compile(r"^(no|nope|incorrect)\b", re.I)
YES_CONTENT = re.compile(
    r"\b(the answer is yes|answer:\s*yes|it is true that)\b", re.I)
NO_CONTENT = re.compile(
    r"\b(the answer is no|answer:\s*no|is not|are not|was not|were not"
    r"|does not|do not|did not|cannot be considered)\b", re.I)


def classify_behavior(text):
    """回答全文を YES / NO / HEDGE に分類（先頭語 → 内容パターンの順）"""
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


def single_token_ids(tokenizer, words):
    ids = set()
    for w in words:
        toks = tokenizer.encode(w, add_special_tokens=False)
        if len(toks) == 1:
            ids.add(toks[0])
    return sorted(ids)


def build_inputs(tokenizer, question, device, no_think=False):
    messages = [{"role": "user", "content": question}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    if no_think:
        # DeepSeek-R1系: 空の思考ブロックをプリフィルして思考をスキップ
        if prompt.rstrip().endswith("<think>"):
            prompt = prompt.rstrip() + "\n\n</think>\n\n"
        else:
            prompt = prompt + "<think>\n\n</think>\n\n"
    return tokenizer(prompt, return_tensors="pt").to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--no-think", action="store_true",
                    help="DeepSeek-R1系: 空のthinkブロックをプリフィルして思考をスキップ")
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_tag = args.model.split("/")[-1]
    outdir = os.path.join(base, "results", model_tag)
    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(base, "questions_v1.yaml")) as f:
        qdata = yaml.safe_load(f)
    questions = [
        (cat, q["id"], q["text"]) for cat in qdata for q in qdata[cat]
    ]
    print(f"{len(questions)} questions / model={args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        qconf = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=qconf, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    final_norm = model.get_decoder().norm
    lm_head = model.get_output_embeddings()
    yes_ids = single_token_ids(tokenizer, YES_WORDS)
    no_ids = single_token_ids(tokenizer, NO_WORDS)

    rows = []
    gen_log = open(os.path.join(outdir, "generations.jsonl"), "w")
    for k, (cat, qid, qtext) in enumerate(questions):
        inputs = build_inputs(tokenizer, qtext, model.device,
                              no_think=args.no_think)

        # --- 読み値（生成前・最終層） + 決断深さ ---
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hs = out.hidden_states
        n_points = len(hs)
        per_layer = []
        for i in range(n_points):
            h = hs[i][0, -1]
            if i < n_points - 1:
                h = final_norm(h)
            logits = lm_head(h.to(lm_head.weight.dtype)).float()
            probs = torch.softmax(logits, dim=-1)
            per_layer.append((float(probs[yes_ids].sum()),
                              float(probs[no_ids].sum())))
        p_yes, p_no = per_layer[-1]
        p_other = 1.0 - p_yes - p_no
        lens_class = ["YES", "NO", "HEDGE"][
            [p_yes, p_no, p_other].index(max(p_yes, p_no, p_other))]
        # 決断深さ: 最終層の勝者が prob>0.5 で勝ち始める最浅の層
        winner_idx = ["YES", "NO", "HEDGE"].index(lens_class)
        decision_layer = n_points - 1
        for i in range(n_points - 1, -1, -1):
            py, pn = per_layer[i]
            vals = [py, pn, 1.0 - py - pn]
            if vals.index(max(vals)) == winner_idx and max(vals) > 0.5:
                decision_layer = i
            else:
                break

        # --- 行動（n回サンプリング） ---
        with torch.no_grad():
            gens = model.generate(
                **inputs, do_sample=True, temperature=args.temperature,
                top_p=0.95, max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.n_samples,
                pad_token_id=tokenizer.eos_token_id)
        plen = inputs["input_ids"].shape[1]
        counts = {"YES": 0, "NO": 0, "HEDGE": 0}
        texts = []
        for g in gens:
            ans = tokenizer.decode(g[plen:], skip_special_tokens=True)
            counts[classify_behavior(ans)] += 1
            texts.append(ans)
        n = args.n_samples
        beh_class = max(counts, key=counts.get)
        row = {
            "cat": cat, "qid": qid, "question": qtext,
            "lens_p_yes": round(p_yes, 4), "lens_p_no": round(p_no, 4),
            "lens_p_other": round(p_other, 4), "lens_class": lens_class,
            "decision_layer": decision_layer, "n_layers": n_points - 1,
            "beh_yes": counts["YES"] / n, "beh_no": counts["NO"] / n,
            "beh_hedge": counts["HEDGE"] / n, "beh_class": beh_class,
            "match": lens_class == beh_class,
        }
        rows.append(row)
        gen_log.write(json.dumps(
            {"qid": qid, "question": qtext, "answers": texts},
            ensure_ascii=False) + "\n")
        print(f"[{k+1:2d}/{len(questions)}] {qid} lens={lens_class:5s} "
              f"beh={beh_class:5s} ({counts}) "
              f"{'OK' if row['match'] else 'MISMATCH'} "
              f"decision@L{decision_layer}/{n_points-1}")
    gen_log.close()

    # --- 集計 ---
    n_match = sum(r["match"] for r in rows)
    acc = n_match / len(rows)
    rho, pval = spearmanr([r["lens_p_other"] for r in rows],
                          [r["beh_hedge"] for r in rows])
    conf = {}
    for r in rows:
        conf.setdefault(r["lens_class"], {}).setdefault(r["beh_class"], 0)
        conf[r["lens_class"]][r["beh_class"]] += 1
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["cat"], []).append(r["match"])

    summary = {
        "model": args.model,
        "n_questions": len(rows),
        "n_samples_per_q": args.n_samples,
        "agreement_acc": round(acc, 3),
        "spearman_p_other_vs_hedge_rate": {
            "rho": round(float(rho), 3), "p": float(pval)},
        "agreement_by_category": {
            c: f"{sum(v)}/{len(v)}" for c, v in by_cat.items()},
        "confusion_lens_vs_behavior": conf,
        "median_decision_layer": sorted(
            r["decision_layer"] for r in rows)[len(rows) // 2],
        "rows": rows,
    }
    with open(os.path.join(outdir, "agreement.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    print(f"\n=== {model_tag} ===")
    print(f"agreement: {n_match}/{len(rows)} = {acc:.1%}")
    print(f"by category: {summary['agreement_by_category']}")
    print(f"Spearman(lens P(other), behavioral hedge rate): "
          f"rho={rho:.3f} (p={pval:.2g})")
    print(f"median decision layer: {summary['median_decision_layer']}"
          f"/{rows[0]['n_layers']}")
    print(f"saved: {os.path.join(outdir, 'agreement.json')}")


if __name__ == "__main__":
    main()
