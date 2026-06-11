# LLM Hidden State Attitude Analysis

Large Language Model (LLM)の隠れ状態解析による推論態度測定の教材

## 概要
対立命題への反応を通じてLLMの断定性・慎重性を可視化する手法の学習教材です。理論学習から実践的な解析まで、段階的に学べるように構成されています。

## 🎯 学習の進め方

### 1. 理論学習：スライド資料で基礎を理解
📊 **[授業用スライド資料](SLIDES.md)**
- LLMの隠れ状態とは何か
- HBDI（Hidden Bias Detection Index）指標の理解
- 実際の測定結果とその解釈
- 各LLMの推論態度の違い
- 【応用編】Hidden State可視化のAIエージェント連携への応用構想（Latent Attitude Consensus Module）

### 2. 実践学習：解析ツールで体験
🔬 **[Hidden State解析ツールキット](hidden_state_analysis/)**

#### 🚀 すぐに始める
- **[環境セットアップ](hidden_state_analysis/setup_and_check.ipynb)** - 必要な依存関係のインストールと動作確認
- **[メイン解析ノートブック](hidden_state_analysis/hidden_state_analysis_notebook.ipynb)** - インタラクティブな解析体験

#### 📋 詳細情報
- **[解析ツールの使い方](hidden_state_analysis/README.md)** - 詳細な説明とトラブルシューティング
- **[パッケージ要件](hidden_state_analysis/requirements.txt)** - 必要なPythonライブラリ

#### 📊 解析結果データ
- **[解析結果画像](hidden_state_analysis/hidden_state_analysis_results.png)** - 4つのグラフによる可視化
- **[HBDI指標データ](hidden_state_analysis/hbdi_scores.csv)** - 各モデルのHBDI値
- **[距離比較データ](hidden_state_analysis/model_distances_comparison.csv)** - 詳細な距離計算結果

### 3. 上級者向け：Hidden State抽出の実装
⚠️ **注意：高性能GPU・大容量メモリが必要**

🛠️ **[Hidden State抽出スクリプト](hidden_state_extraction/)**
- **[メイン抽出スクリプト](hidden_state_extraction/extract_hidden_states.py)** - LLMからHidden Stateを取得
- **[環境構築マニュアル](hidden_state_extraction/README.md)** - 実行環境の詳細設定
- **[抽出済みデータ](hidden_state_extraction/results/)** - DeepSeek、LLaMA、QwenのHidden State

## 📁 プロジェクト構成

```
llm-hidden-state-attitude-analysis/
├── README.md                     # このファイル
├── LICENSE                       # ライセンス情報
├── SLIDES.md                     # 📊 授業用スライド資料
├── hidden_state_analysis/        # 🔬 解析ツールキット
│   ├── setup_and_check.ipynb         # 環境セットアップ
│   ├── hidden_state_analysis_notebook.ipynb  # メイン解析
│   ├── requirements.txt              # 依存パッケージ
│   ├── README.md                     # 詳細説明
│   ├── hidden_state_analysis_results.png     # 解析結果画像
│   ├── hbdi_scores.csv                       # HBDI指標
│   └── model_distances_comparison.csv        # 距離データ
└── hidden_state_extraction/      # 🛠️ Hidden State抽出
    ├── extract_hidden_states.py     # メイン抽出スクリプト
    ├── README.md                    # 環境構築マニュアル・フォルダガイド
    └── results/                     # 抽出済みデータ
        ├── deepseek_hidden_state.csv
        ├── llama_hidden_state.csv
        └── qwen_hidden_state.csv
```

## 🎓 学習目標

この教材を通じて以下を学習できます：

1. **理論理解**
   - LLMの内部状態（Hidden State）の概念
   - 対立命題テストによる態度測定手法
   - HBDI指標の計算方法と解釈

2. **実践スキル**
   - Jupyter Notebookを使った データ解析
   - コサイン距離による類似度計算
   - 機械学習結果の可視化技術

3. **批判的思考**
   - LLMの推論プロセスの多様性理解
   - AI安全性・バイアス検出の重要性
   - 適切なモデル選択の判断基準

## 🔧 必要な環境

### 基本学習（スライド＋解析ツール）
- Python 3.7+
- Jupyter Notebook/Lab
- 標準的なデータ分析ライブラリ（pandas, numpy, matplotlib等）

### 上級学習（Hidden State抽出）
- 高性能GPU（VRAM 8GB以上推奨）
- CUDA対応環境
- 大容量メモリ（16GB以上推奨）

## 📚 関連情報

- **対象レベル**: 機械学習の基礎知識を持つ学習者
- **推奨学習時間**: 2-4時間（解析実習含む）
- **ライセンス**: [LICENSE](LICENSE)参照

## 🤝 貢献・フィードバック

改善提案やバグレポートは、GitHubのIssuesまでお願いします。
