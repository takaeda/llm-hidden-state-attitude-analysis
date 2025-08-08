# Hidden State Analysis

このディレクトリには、LLMのHidden State データを解析し、対立命題に対する各モデルの態度を比較するためのJupyter Notebookが含まれています。

## 概要

このツールは以下の解析を行います：

1. **コサイン距離計算**: 各対立ペア間のHidden State距離を計算
2. **HBDI指標計算**: Hidden Bias Detection Index（政治的質問距離 / 自然科学平均距離）
3. **モデル比較**: DeepSeek, LLaMA, Qwenの態度を比較
4. **可視化**: 4つのグラフで結果を表示
5. **要約レポート**: 各モデルの特徴をテキストで要約

## 対象データ

- **政治的質問**: Q1（台湾の地位に関する質問）
- **自然科学質問**: Q2-Q10（地球の公転、水の沸点、DNA構造など）

## 🚀 実行方法（重要）

### ステップ1：事前準備（必須）

**まず `setup_and_check.ipynb` を実行してください：**

1. Jupyter Notebook/Lab で `setup_and_check.ipynb` を開く
2. セルを上から順番にすべて実行
3. 全てのチェックが✅になることを確認

### ステップ2：メイン解析

準備完了後、`hidden_state_analysis_notebook.ipynb` を実行：

1. Jupyter Notebook/Lab で `hidden_state_analysis_notebook.ipynb` を開く
2. セルを上から順番に実行
3. 各段階の結果を確認しながら学習

## 必要な環境

### 依存パッケージ

事前準備ノートブックで自動インストールされますが、手動の場合：

```bash
pip install -r requirements.txt
```

必要なパッケージ：
- pandas (データ処理)
- numpy (数値計算) 
- matplotlib (基本グラフ)
- seaborn (統計グラフ)
- scikit-learn (コサイン距離計算)

## 出力結果

解析ノートブックは以下のファイルを生成します：

- `hidden_state_analysis_results.png`: 4つのグラフを含む可視化結果
- `model_distances_comparison.csv`: 各モデル・質問ペアの距離データ
- `hbdi_scores.csv`: HBDI指標とその構成要素
- Notebook内出力: 詳細な要約レポート

## 出力の解釈

### HBDI指標の意味

- **< 0.6**: 慎重・分析的（適切な慎重さ）
- **≈ 1.0**: バランス型（科学的事実と同程度の区別）
- **> 1.2**: 断定的・確信的（要注意の断定性）

### 期待される結果パターン

1. **慎重型LLM**: 政治的質問でも慎重に対応 → 小さなHBDI
2. **断定型LLM**: 政治的質問で明確に判断 → 大きなHBDI

## 生成される可視化

1. **棒グラフ**: 全質問ペアの距離比較
2. **散布図**: 政治 vs 自然科学距離の関係
3. **HBDI比較**: 各モデルのHBDI指標
4. **箱ひげ図**: 自然科学質問の距離分布

## ファイル構造

```
hidden_state_analysis/
├── setup_and_check.ipynb       # 事前準備・環境確認ノートブック
├── hidden_state_analysis_notebook.ipynb  # メイン解析ノートブック
├── requirements.txt             # 必要パッケージリスト
├── README.md                   # このファイル
├── hidden_state_analysis_results.png  # 実行後生成
├── model_distances_comparison.csv     # 実行後生成
└── hbdi_scores.csv                    # 実行後生成
```

## トラブルシューティング

### データファイルが見つからない場合

```
Error: No data files found!
```

以下を確認：
1. `../hidden_state_extraction/results/` にCSVファイルが存在するか
2. ファイル名が `deepseek_hidden_state.csv`, `llama_hidden_state.csv`, `qwen_hidden_state.csv` であるか

### 実行時エラー

依存パッケージが不足している場合：
```bash
pip install --upgrade pandas numpy matplotlib seaborn scikit-learn
```

## 注意事項

- CSVファイルは大容量（4096次元ベクトル）のため、読み込みに時間がかかる場合があります
- メモリ使用量が大きくなる可能性があります
- グラフの表示には適切なGUI環境が必要です（Jupyter Notebook推奨）

## 研究・教育利用について

このツールは教育目的で作成されており、以下の学習に活用できます：

- LLMの内部状態の理解
- 機械学習モデルの態度・バイアス分析
- コサイン距離を用いた類似度解析
- データ可視化手法の実践
