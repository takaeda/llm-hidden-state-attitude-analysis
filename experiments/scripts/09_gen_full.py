#!/usr/bin/env python3
"""検証(a)用: 完全な回答文を n回生成し、テキストと出だし3トークンの
hidden stateの両方を保存する。後段で「hidden stateの広がり」と
「(別モデルで測る)意味的な広がり」が一致するかを照合する。
"""
import argparse
import json
import os

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=48)
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
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kw["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()
    eos = tok.eos_token_id

    texts_out = {}
    vec_store = {}
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
        texts, vecs = [], []
        for gi in range(args.n):
            gen_ids = gens[gi][plen:]
            keep = (gen_ids != eos).nonzero()
            L = int(keep[-1])+1 if len(keep) else 1
            texts.append(tok.decode(gen_ids[:L], skip_special_tokens=True))
            seq = gens[gi][:plen+3].unsqueeze(0)
            with torch.no_grad():
                out = model(seq, output_hidden_states=True)
            hs = out.hidden_states[-1][0, plen:plen+3].float().cpu().numpy()
            vecs.append(hs.mean(0))
        texts_out[qid] = {"category": cat, "question": qtext, "texts": texts}
        vec_store[qid] = np.vstack(vecs)
        print(f"[{ki+1:2d}/36] {qid} {cat:13s} generated {args.n} answers")

    json.dump(texts_out, open(os.path.join(outdir, "full_texts.json"), "w"),
              ensure_ascii=False, indent=1)
    np.savez(os.path.join(outdir, "full_first3.npz"), **vec_store)
    print(f"saved: {outdir}/full_texts.json, full_first3.npz")


if __name__ == "__main__":
    main()
