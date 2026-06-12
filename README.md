# LLM Hidden State Analysis ― 教材と再現コード

AIの「頭の中」（hidden state）を覗くと何が見えるか。**出力の文章を読まずにLocal LLMの一貫性を評価する**という応用までを、入門と応用の二層で扱う教材です。

## 🎯 教材は二層構成です

📘 **[LECTURE_hidden_state.md](LECTURE_hidden_state.md)** ― **本編（入門・全10スライド）**

予備知識ゼロで読める講義資料。「同じ質問なのにAIの答えがゆれるのはなぜ？」という誰もが体験した謎から始め、サイコロの鎖 → 実際に12回聞いてみる → 頭の中を点にして地図で見る（toorPIA）→ 一貫性≠正しさ → AIの得意な土俵マップ → AI合議の夢、までを実物の実験だけで進みます。数式・専門用語は使いません。

📗 **[ADVANCED.md](ADVANCED.md)** ― **応用資料（全19スライド）**

本編の内容を研究レベルまで深掘り：測定の設計（出だし数語の窓・アンカー規格化・d′安全弁）、検証（意味的一貫性との相関）、専門的Q&A（temperature=0でよいのでは？／木展開で直接計算できるのでは？／どの層で測るべきか？）。本編で「自分もやりたい」と思った人の次の一歩。

## 🔬 再現する

📂 **[experiments/](experiments/)** ― 全ての図・数値を実際に追試するためのコードと結果サマリ

- 測定スクリプト・質問/アンカー定義・要件・測定済み結果JSON・図の生成スクリプト
- 手順とスクリプト→図の対応表は [`experiments/README.md`](experiments/README.md)
- GPUがあれば自分のモデルで再測定、無ければ収録済みデータで集計・可視化を再現できる

## 📁 構成

```
llm-hidden-state-attitude-analysis/
├── LECTURE_hidden_state.md   # 📘 本編（入門）
├── ADVANCED.md               # 📗 応用資料（研究レベル）
├── images/                   # 図（e0*=本編用 / 0*=応用資料用）
├── experiments/              # 🔬 再現コード + 結果サマリ
└── SLIDES.md                 # 〔旧版・参照用〕対立命題テスト/HBDI指標の詳説
```

## 🗂️ 旧版について

[`SLIDES.md`](SLIDES.md) は昨年度の「対立命題テスト・HBDI指標」版の資料です。その考え方（対立ペア・出だし数語・モデル内正規化）は ADVANCED.md に引き継がれています。旧版に付随していた解析ノートブック（`hidden_state_analysis/`）と抽出スクリプト（`hidden_state_extraction/`）は、`experiments/` のパイプラインに置き換えられたため削除しました。

## 📜 ライセンス

Copyright (c) 2026 toor Inc.

本教材（文書・コード・図）は [MITライセンス](LICENSE) で提供します。
