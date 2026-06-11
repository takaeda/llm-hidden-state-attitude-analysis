# LLM Hidden State Analysis ― 教材と再現コード

LLMの内部状態（hidden state）が何を表しているかをゼロから理解し、**出力文を読まずにLocal LLMの出力一貫性を評価する**ところまでを扱う教材です。

## 🎯 主教材

📘 **[LECTURE_hidden_state.md](LECTURE_hidden_state.md)** ― 白紙から理解する本編（全16スライド）

予備知識ゼロから次の順に積み上げます：
hidden stateとは何か → LLMの生成の仕組み → 出力がゆらぐという実務の壁 → ゆらぎの出所（鎖のイメージ）→ 状態が表す「意味」「次の語」「確信度」→ 何度も答えさせて散らばりを測る → toorPIAで可視化 → 一貫性≠正しさ → 物差し（アンカー規格化）と安全弁(d′) → 出力一貫性によるLocal LLM品質評価 → 検証 → 複数LLM合議への応用。

図はすべて [`images/`](images/) に収録。

## 🔬 再現する

📂 **[experiments/](experiments/)** ― 本編の図・数値を実際に追試するためのコードと結果サマリ

- スクリプト（`logit_lens.py`, `08_consistency_metric.py`, `09_gen_full.py`+`10_validate.py` 等）と、質問・アンカー定義、計画書、要件、測定済みの結果JSON
- 手順とスクリプト→図の対応表は [`experiments/README.md`](experiments/README.md)
- GPUがあれば自分のモデルで再測定、無ければ収録済みの数値で集計・可視化を再現できる

## 📁 構成

```
llm-hidden-state-attitude-analysis/
├── LECTURE_hidden_state.md   # 📘 主教材（本編）
├── images/                   # 本編の図
├── experiments/              # 🔬 再現コード + 結果サマリ + 計画書
└── SLIDES.md                 # 〔旧版・参照用〕対立命題テスト/HBDI指標の詳説
```

## 🗂️ 旧版について

[`SLIDES.md`](SLIDES.md) は昨年度の「対立命題テスト・HBDI指標」版の資料です。本編 `LECTURE_hidden_state.md` はこれを発展・再構成したもので、HBDIの考え方（対立ペア・出だし数語・モデル内正規化）は本編にも引き継がれています。旧版に付随していた解析ノートブック（`hidden_state_analysis/`）と抽出スクリプト（`hidden_state_extraction/`）は、`experiments/` のパイプラインに置き換えられたため削除しました。
