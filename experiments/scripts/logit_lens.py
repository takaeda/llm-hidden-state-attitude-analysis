#!/usr/bin/env python3
"""Logit Lens demo: 各層のhidden stateをモデル自身のunembedding（lm_head）で
語彙空間に射影し、「その層の時点でモデルが次に言おうとしている語」の
層方向の推移を可視化する。

プロンプト（質問）末尾トークン位置の全層hidden stateを対象とする。
= 1語も生成していない時点で、回答の方向性（Yes / No / ヘッジ）が
  どの深さでどう競り合い、どこで決着するかを見る。

usage:
  python logit_lens.py --model Qwen/Qwen3-0.6B
  python logit_lens.py --model Qwen/Qwen3-4B --load-4bit
"""

import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

QUESTIONS = [
    ("science_true", "Does the Earth revolve around the Sun?"),
    ("science_false", "Does the Sun revolve around the Earth?"),
    ("political", "Is Taiwan an independent country?"),
    ("contested", "Is nuclear power the best way to fight climate change?"),
]

# 回答の方向性を代表するトークン群（単一トークンになるもののみ採用）
YES_WORDS = ["Yes", " Yes", "yes", " yes"]
NO_WORDS = ["No", " No", "no", " no"]
HEDGE_WORDS = [
    "It", " It", "This", " This", "The", " The", "That", " That",
    "As", " As", "While", " While", "Whether", " Whether", "Well", " Well",
]


def single_token_ids(tokenizer, words):
    ids = set()
    for w in words:
        toks = tokenizer.encode(w, add_special_tokens=False)
        if len(toks) == 1:
            ids.add(toks[0])
    return sorted(ids)


def build_prompt(tokenizer, question, no_think=False):
    messages = [{"role": "user", "content": question}]
    try:
        # Qwen3: 思考モードを無効化して「素の応答開始点」を見る
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    if no_think:
        # DeepSeek-R1系: 空の思考ブロックをプリフィルして思考をスキップし、
        # 素の回答開始点のhidden stateを測る
        if prompt.rstrip().endswith("<think>"):
            prompt = prompt.rstrip() + "\n\n</think>\n\n"
        else:
            prompt = prompt + "<think>\n\n</think>\n\n"
    return prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--no-think", action="store_true",
                    help="DeepSeek-R1系: 空のthinkブロックをプリフィルして思考をスキップ")
    ap.add_argument("--gpu-mem", default=None,
                    help="GPUに割り当てる最大メモリ(例 5GiB)。残りはCPUへオフロード")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--gen-tokens", type=int, default=30,
                    help="参考として実際の回答冒頭を生成するトークン数")
    args = ap.parse_args()

    model_tag = args.model.split("/")[-1]
    outdir = args.outdir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", model_tag,
    )
    os.makedirs(outdir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    load_kwargs = {"device_map": "auto"}
    if args.gpu_mem:
        load_kwargs["max_memory"] = {0: args.gpu_mem, "cpu": "48GiB"}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            # GPUに収まらない層をCPUへ置く場合に必要
            llm_int8_enable_fp32_cpu_offload=bool(args.gpu_mem),
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()

    # 最終正規化層と unembedding（logit lens の標準構成）
    final_norm = model.get_decoder().norm
    lm_head = model.get_output_embeddings()

    yes_ids = single_token_ids(tokenizer, YES_WORDS)
    no_ids = single_token_ids(tokenizer, NO_WORDS)
    hedge_ids = single_token_ids(tokenizer, HEDGE_WORDS)
    print(f"tracked token ids: yes={yes_ids} no={no_ids} hedge={len(hedge_ids)} ids")

    all_results = {}
    for qid, question in QUESTIONS:
        prompt = build_prompt(tokenizer, question, no_think=args.no_think)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        # hidden_states: (embedding出力, 第1層出力, ..., 第N層出力(最終normは適用済み))
        hs = out.hidden_states
        n_points = len(hs)

        layers = []
        for i in range(n_points):
            h = hs[i][0, -1]  # プロンプト最終トークン位置
            if i < n_points - 1:
                h = final_norm(h)  # 中間層には最終normを適用（標準的logit lens）
            logits = lm_head(h.to(lm_head.weight.dtype)).float()
            probs = torch.softmax(logits, dim=-1)

            ent = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
            top = torch.topk(probs, 5)
            top_tokens = [
                (tokenizer.decode([t]), float(p))
                for t, p in zip(top.indices.tolist(), top.values.tolist())
            ]
            layers.append({
                "layer": i,
                "p_yes": float(probs[yes_ids].sum()),
                "p_no": float(probs[no_ids].sum()),
                "p_hedge": float(probs[hedge_ids].sum()),
                "entropy_nats": ent,
                "top5": top_tokens,
            })

        # 参考: 実際の回答冒頭（貪欲法）
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=args.gen_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(
            gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )

        all_results[qid] = {
            "question": question,
            "greedy_answer_head": answer,
            "layers": layers,
        }
        print(f"\n=== {qid}: {question}")
        print(f"  greedy answer head: {answer!r}")
        last = layers[-1]
        print(f"  final layer: P(yes)={last['p_yes']:.3f} P(no)={last['p_no']:.3f} "
              f"P(hedge)={last['p_hedge']:.3f} H={last['entropy_nats']:.2f} nats")
        print(f"  final top5: {last['top5']}")

    with open(os.path.join(outdir, "logit_lens.json"), "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=1)

    # ---- 可視化: 質問ごとに P(yes)/P(no)/P(hedge) + エントロピーの層推移 ----
    n_q = len(QUESTIONS)
    fig, axes = plt.subplots(1, n_q, figsize=(5.2 * n_q, 4.4), sharey=True)
    vocab = lm_head.weight.shape[0]
    max_ent = math.log(vocab)
    for ax, (qid, _) in zip(axes, QUESTIONS):
        r = all_results[qid]
        xs = [l["layer"] for l in r["layers"]]
        ax.plot(xs, [l["p_yes"] for l in r["layers"]], label="P(Yes)", color="tab:green", lw=2)
        ax.plot(xs, [l["p_no"] for l in r["layers"]], label="P(No)", color="tab:red", lw=2)
        ax.plot(xs, [l["p_hedge"] for l in r["layers"]], label="P(hedge starters)",
                color="tab:blue", lw=2)
        ax.plot(xs, [l["entropy_nats"] / max_ent for l in r["layers"]],
                label="entropy (norm.)", color="gray", ls="--", lw=1.5)
        ax.set_title(f"{qid}\n{r['question']}", fontsize=9)
        ax.set_xlabel("layer")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
        ans = r["greedy_answer_head"].replace("\n", " ")[:60]
        ax.text(0.02, -0.32, f"answer: {ans}", transform=ax.transAxes, fontsize=7.5)
    axes[0].set_ylabel("probability of next token")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Logit Lens at prompt-end position — {model_tag}", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    png = os.path.join(outdir, "logit_lens.png")
    fig.savefig(png, dpi=140)
    print(f"\nsaved: {png}")
    print(f"saved: {os.path.join(outdir, 'logit_lens.json')}")


if __name__ == "__main__":
    main()
