#!/usr/bin/env python3
"""出力文の「全トークン」の最終層hidden stateを平均せず1点ずつ保存する（PLAN3）。

これまで（03/06/09）は出だし数トークンを平均して1試行=1点にしていた。本スクリプトは
平均をやめ、生成区間の各トークン位置の hidden state をそのまま全部書き出す。
後段 figures/fig_traj.py が toorPIA に投入し、生成順に線で結んで軌跡の形を見る。

各トークン行のメタ: sent_id, gen_mode(greedy/sample), variant, tok_idx, token,
  is_boundary(句読点・改行・文末), is_anchor(生成直前のプロンプト末尾なら1), norm(L2)

usage:
  # Phase 0: 1プロンプトを greedy で
  python 15_trajectory.py --model Qwen/Qwen3-4B --load-4bit --no-think --subset p1_earth
  # Phase 1: 全プロンプト greedy
  python 15_trajectory.py --model Qwen/Qwen3-4B --load-4bit --no-think
  # Phase 2: サンプリングで形のばらつき
  python 15_trajectory.py --model Qwen/Qwen3-4B --load-4bit --no-think --sample --n-sample 4
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

META_COLS = ["sent_id", "gen_mode", "variant", "tok_idx",
             "token", "is_boundary", "is_anchor", "norm"]


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
    ap.add_argument("--sample", action="store_true",
                    help="付けるとサンプリング生成（既定は greedy=決定的）")
    ap.add_argument("--n-sample", type=int, default=4)
    ap.add_argument("--gen-batch", type=int, default=0,
                    help="サンプリング生成を何本ずつに分けるか（0=一括）。VRAM節約用。例: 25")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--min-new-tokens", type=int, default=0,
                    help=">0で最低生成長を強制（1語で停止するのを防ぐ）")
    ap.add_argument("--subset", default=None, help="カンマ区切りの sent_id。省略時は全部")
    ap.add_argument("--prompts", default=None, help="プロンプトyaml。省略時は traj_prompts.yaml")
    ap.add_argument("--layer", type=int, default=-1,
                    help="抽出する層（hidden_states索引）。-1=最終層。中間層は例: 18")
    ap.add_argument("--suffix", default="",
                    help="出力ファイル名の接尾辞（層別保存用。例: _L18）")
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompts_path = args.prompts or os.path.join(base, "traj_prompts.yaml")
    prompts = yaml.safe_load(open(prompts_path))
    if args.subset:
        keep = set(args.subset.split(","))
        prompts = [p for p in prompts if p["id"] in keep]

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

    gen_mode = "sample" if args.sample else "greedy"
    n_var = args.n_sample if args.sample else 1

    rows = []          # CSV 行（メタ + 2560次元）
    meta_records = []  # traj_meta.json
    dim = None

    # サンプリングを何本ずつに割るか（VRAM節約）
    bsz = args.gen_batch if (args.sample and args.gen_batch > 0) else n_var
    batches = []
    rem = n_var
    while rem > 0:
        batches.append(min(bsz, rem)); rem -= batches[-1]

    for pi, p in enumerate(prompts):
        sid, qtext = p["id"], p["text"]
        inputs = build_inputs(tok, qtext, model.device, args.no_think)
        plen = inputs["input_ids"].shape[1]
        v = 0
        for csz in batches:
            gkw = {"max_new_tokens": args.max_new_tokens, "pad_token_id": eos}
            if args.min_new_tokens > 0:
                gkw["min_new_tokens"] = args.min_new_tokens
            with torch.no_grad():
                if args.sample:
                    gens = model.generate(
                        **inputs, do_sample=True, temperature=args.temperature,
                        top_p=0.95, num_return_sequences=csz, **gkw)
                else:
                    gens = model.generate(
                        **inputs, do_sample=False, num_return_sequences=1, **gkw)

            for vl in range(gens.shape[0]):
                seq = gens[vl]
                gen_ids = seq[plen:]
                keep = (gen_ids != eos).nonzero()
                L = int(keep[-1]) + 1 if len(keep) else 1
                full = seq[: plen + L].unsqueeze(0)
                with torch.no_grad():
                    out = model(full, output_hidden_states=True)
                hs = out.hidden_states[args.layer][0].float().cpu().numpy()  # (plen+L, dim)
                if dim is None:
                    dim = hs.shape[1]

                # 開始アンカー: 生成直前=プロンプト末尾位置
                anchor = hs[plen - 1]
                rows.append(_row(sid, gen_mode, v, -1, "<start>", 0, 1, anchor))
                # 生成区間の全トークン
                for t in range(L):
                    vec = hs[plen + t]
                    tok_str = tok.decode([int(gen_ids[t])])
                    rows.append(_row(sid, gen_mode, v, t, tok_str,
                                     int(is_boundary_token(tok_str)), 0, vec))

                text = tok.decode(gen_ids[:L], skip_special_tokens=True)
                meta_records.append({"sent_id": sid, "kind": p.get("kind", ""),
                                     "gen_mode": gen_mode, "variant": v,
                                     "prompt": qtext, "n_tokens": L, "text": text})
                if v % 10 == 0 or not args.sample:
                    print(f"[{pi+1}/{len(prompts)}] {sid} {gen_mode} v{v}: {L} tokens | {text[:60]!r}")
                v += 1

    # 書き出し（tokenはカンマ等を含むので csv.writer で安全にクォート）
    csv_path = os.path.join(outdir, f"traj_vectors{args.suffix}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(META_COLS + [f"d{i}" for i in range(dim)])
        for r in rows:
            w.writerow(r)
    json.dump(meta_records, open(os.path.join(outdir, f"traj_meta{args.suffix}.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\nsaved: {csv_path} ({len(rows)} token-points, dim={dim})")
    print(f"saved: {os.path.join(outdir, f'traj_meta{args.suffix}.json')} ({len(meta_records)} sentences)")


def _row(sid, mode, v, tok_idx, token, is_b, is_a, vec):
    norm = float(np.linalg.norm(vec))
    return ([sid, mode, v, tok_idx, token, is_b, is_a, f"{norm:.5f}"]
            + [f"{x:.5f}" for x in vec])


if __name__ == "__main__":
    main()
