#!/usr/bin/env python3
"""検証(a): 「出だし3語のhidden stateの広がり」が、
「(独立した意味埋め込みモデルで測る)回答文の意味的な広がり」と一致するかを照合する。

- hidden側: full_first3.npz の各質問のベクトル群の平均ペアワイズ・コサイン距離
- 意味側  : full_texts.json の各回答を sentence-transformer で埋め込み、その広がり
両者を全質問で相関（Spearman）。高ければ「読まずに一貫性を測れる」が成立。
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = sys.argv[1:] or ["Qwen3-4B", "Mistral-7B-Instruct-v0.3"]
EMB = "sentence-transformers/all-mpnet-base-v2"


def spread(vs):
    if len(vs) < 2:
        return 0.0
    vn = vs/(np.linalg.norm(vs, axis=1, keepdims=True)+1e-9)
    S = vn@vn.T
    return float(1-S[np.triu_indices(len(vs), 1)].mean())


emb_model = SentenceTransformer(EMB)
result = {}
for tag in MODELS:
    d = os.path.join(BASE, "results", tag)
    texts = json.load(open(os.path.join(d, "full_texts.json")))
    vecs = np.load(os.path.join(d, "full_first3.npz"))
    rows = []
    for qid in texts:
        cat = texts[qid]["category"]
        hid = spread(vecs[qid])                      # hidden側の広がり
        emb = emb_model.encode(texts[qid]["texts"], normalize_embeddings=True)
        sem = spread(emb)                            # 意味側の広がり
        rows.append({"qid": qid, "category": cat,
                     "hidden_spread": round(hid, 5), "semantic_spread": round(sem, 5)})
    hs = [r["hidden_spread"] for r in rows]
    ss = [r["semantic_spread"] for r in rows]
    rho, p = spearmanr(hs, ss)
    pear = float(np.corrcoef(hs, ss)[0, 1])
    result[tag] = {"spearman": round(float(rho), 3), "p_value": float(p),
                   "pearson": round(pear, 3), "rows": rows}
    print(f"\n=== {tag} ===")
    print(f"Spearman(hidden広がり, 意味広がり) = {rho:.3f} (p={p:.2g})  Pearson={pear:.3f}")
    # 両端の例を表示（読まずに測れているかの定性確認用）
    sr = sorted(rows, key=lambda r: r["hidden_spread"])
    print(f"  最も一貫(hidden小): {sr[0]['qid']} hidden={sr[0]['hidden_spread']} 意味={sr[0]['semantic_spread']}")
    print(f"  最もバラつく(hidden大): {sr[-1]['qid']} hidden={sr[-1]['hidden_spread']} 意味={sr[-1]['semantic_spread']}")

json.dump(result, open(os.path.join(BASE, "results", "validation_a.json"), "w"),
          ensure_ascii=False, indent=1)
print(f"\nsaved: results/validation_a.json")
