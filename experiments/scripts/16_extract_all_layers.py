#!/usr/bin/env python3
"""全対象層の「全トークン hidden state」を1回の生成パスで抽出し、層別 npy で保存する。

層スイープ用(17_)と同じ単一 forward(output_hidden_states=True) で全層を取得し、層ごとに
生成区間の全トークン(smooth5)を保持して `quilt/vec_L{n}.npy`(float16) に保存。
全層は同一の生成系列に由来するので、後段の toorPIA キルト図は層間で厳密に比較可能。

出力:
  results/<tag>/quilt/vec_L{n}.npy   各層の (Ntok, dim) float16（smooth5後の生成区間トークン）
  results/<tag>/quilt/meta.npz       levels(ndoc,), sizes(ndoc,), layers(list)

usage:
  python scripts/16_extract_all_layers.py --model Qwen/Qwen3-4B --load-4bit --no-think \
    --sample --n-sample 40 --gen-batch 20 --temperature 1.0 --max-new-tokens 90 \
    --prompts stance_prompts.yaml
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"))
from fig_traj import moving_average  # noqa: E402

LEVEL = {"s0_strong_anti": 0, "s1_anti": 1, "s2_neutral": 2, "s3_pro": 3, "s4_strong_pro": 4}


def is_boundary_token(tok):
    s = tok.strip()
    if s == "":
        return "\n" in tok
    return ("\n" in tok) or any(ch in s for ch in ".!?,;:")


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
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--n-sample", type=int, default=40)
    ap.add_argument("--gen-batch", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=90)
    ap.add_argument("--min-new-tokens", type=int, default=0)
    ap.add_argument("--prompts", default="stance_prompts.yaml")
    ap.add_argument("--layers", default="2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36")
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--subset", default="", help="カンマ区切りの sent_id に限定（例 s0_strong_anti,s4_strong_pro）")
    ap.add_argument("--outdir-name", default="quilt", help="results/<tag>/ 下の出力フォルダ名")
    ap.add_argument("--save-tokens", action="store_true", help="トークン文字列・is_boundary も保存")
    ap.add_argument("--save-texts", action="store_true", help="各回答の全文・グループを docs.json に保存")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompts_path = args.prompts if os.path.isabs(args.prompts) else os.path.join(base, args.prompts)
    prompts = yaml.safe_load(open(prompts_path))
    if args.subset:
        keep = set(args.subset.split(","))
        prompts = [p for p in prompts if p["id"] in keep]
    tag = args.model.split("/")[-1]
    outdir = os.path.join(base, "results", tag, args.outdir_name)
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
    nlayer = model.config.num_hidden_layers
    print(f"model layers={nlayer} (hidden_states={nlayer + 1}個), 抽出層={layers}")
    assert max(layers) <= nlayer

    n_var = args.n_sample if args.sample else 1
    bsz = args.gen_batch if (args.sample and args.gen_batch > 0) else n_var
    batches = []
    rem = n_var
    while rem > 0:
        batches.append(min(bsz, rem)); rem -= batches[-1]

    per_layer = {L: [] for L in layers}
    levels, sizes, groups, texts = [], [], [], []
    tok_strs, tok_isb = [], []
    group_names = []   # sent_id を出現順にグループ化（任意プロンプトに汎用）

    for pi, p in enumerate(prompts):
        sid, qtext = p["id"], p["text"]
        if sid not in group_names:
            group_names.append(sid)
        gidx = group_names.index(sid)
        inputs = build_inputs(tok, qtext, model.device, args.no_think)
        plen = inputs["input_ids"].shape[1]
        for csz in batches:
            gkw = {"max_new_tokens": args.max_new_tokens, "pad_token_id": eos}
            if args.min_new_tokens > 0:
                gkw["min_new_tokens"] = args.min_new_tokens
            with torch.no_grad():
                if args.sample:
                    gens = model.generate(**inputs, do_sample=True, temperature=args.temperature,
                                          top_p=0.95, num_return_sequences=csz, **gkw)
                else:
                    gens = model.generate(**inputs, do_sample=False, num_return_sequences=1, **gkw)
            for vl in range(gens.shape[0]):
                seq = gens[vl]
                gen_ids = seq[plen:]
                keep = (gen_ids != eos).nonzero()
                Lk = int(keep[-1]) + 1 if len(keep) else 1
                full = seq[: plen + Lk].unsqueeze(0)
                with torch.no_grad():
                    out = model(full, output_hidden_states=True)
                for L in layers:
                    hs = out.hidden_states[L][0, plen:plen + Lk].float().cpu().numpy()
                    sm = moving_average(hs, args.smooth).astype(np.float16)
                    per_layer[L].append(sm)
                levels.append(LEVEL.get(sid, gidx)); sizes.append(int(Lk)); groups.append(gidx)
                if args.save_tokens:
                    for t in range(Lk):
                        s = tok.decode([int(gen_ids[t])])
                        tok_strs.append(s); tok_isb.append(int(is_boundary_token(s)))
                if args.save_texts:
                    texts.append(tok.decode(gen_ids[:Lk], skip_special_tokens=True).strip())
                del out
        print(f"[{pi+1}/{len(prompts)}] {sid}: {n_var} samples")
        torch.cuda.empty_cache()

    for L in layers:
        vec = np.vstack(per_layer[L]).astype(np.float16)
        np.save(os.path.join(outdir, f"vec_L{L}.npy"), vec)
        per_layer[L] = None
        print(f"saved vec_L{L}.npy  shape={vec.shape}")
    np.savez(os.path.join(outdir, "meta.npz"),
             levels=np.array(levels), sizes=np.array(sizes), layers=np.array(layers),
             groups=np.array(groups), group_names=np.array(group_names, dtype=object))
    if args.save_tokens:
        np.savez(os.path.join(outdir, "tokens.npz"),
                 token=np.array(tok_strs, dtype=object), is_boundary=np.array(tok_isb))
        print(f"saved tokens.npz  ntok={len(tok_strs)}")
    if args.save_texts:
        json.dump([{"group": group_names[groups[i]], "text": texts[i]} for i in range(len(texts))],
                  open(os.path.join(outdir, "docs.json"), "w"), ensure_ascii=False, indent=1)
        print(f"saved docs.json  ndoc={len(texts)}  groups={group_names}")
    print(f"saved meta.npz  ndoc={len(levels)}  outdir={outdir}")


if __name__ == "__main__":
    main()
