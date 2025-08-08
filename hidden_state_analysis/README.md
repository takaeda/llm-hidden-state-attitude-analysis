# Hidden State Analysis

このディレクトリには、LLMのHidden Stateデータを解析するためのJupyter Notebookとツールが含まれています。

## 📂 このフォルダの目的

`../hidden_state_extraction/results/`にある既存のHidden StateデータをもとにHBDI指標を計算し、各モデルの推論態度を可視化・比較します。

**理論的背景やHBDI指標の詳細については[SLIDES.md](../SLIDES.md)を参照してください。**

## 📁 ファイル構成

```
hidden_state_analysis/
├── setup_and_check.ipynb          # 【STEP1】環境セットアップ・動作確認
├── hidden_state_analysis_notebook.ipynb  # 【STEP2】メイン解析実行
├── requirements.txt                # 必要なPythonパッケージ
├── README.md                      # このファイル
├── hidden_state_analysis_results.png     # 【出力】可視化結果
├── model_distances_comparison.csv        # 【出力】距離データ
└── hbdi_scores.csv                       # 【出力】HBDI指標
```

## 🚀 実行手順

### **STEP1: 環境準備**
```bash
# Jupyter Notebookを開いて以下を実行
1. setup_and_check.ipynb を開く
2. 全てのセルを上から順番に実行
3. 全てのチェックが✅になることを確認
```

### **STEP2: メイン解析**
```bash
# 準備完了後
1. hidden_state_analysis_notebook.ipynb を開く
2. 全てのセルを上から順番に実行
3. 結果を確認
```

## 📊 生成される出力ファイル

| ファイル名 | 内容 | 用途 |
|------------|------|------|
| `hidden_state_analysis_results.png` | 4つのグラフによる可視化 | 結果の確認・プレゼンテーション |
| `model_distances_comparison.csv` | 各質問ペアの距離データ | 詳細なデータ分析 |
| `hbdi_scores.csv` | 各モデルのHBDI値 | 定量的評価 |

### 可視化内容（4つのグラフ）
1. **モデル別距離比較**: 各モデルの質問ペアに対する反応
2. **HBDI指標比較**: モデル間の態度の違い（断定的 vs 慎重）

## 💻 動作要件

### 必要な環境
- Python 3.7+
- Jupyter Notebook/Lab
- メモリ: 2GB以上（Hidden Stateデータは4096次元）

### 依存パッケージ
```
pandas >= 1.5.0
numpy >= 1.21.0
matplotlib >= 3.5.0
seaborn >= 0.11.0
scikit-learn >= 1.1.0
```

## 🔧 トラブルシューティング

### データファイルが見つからない場合
```
Error: No data files found!
```
**解決方法**: 以下を確認
- `../hidden_state_extraction/results/`にCSVファイルが存在するか
- ファイル名が正しいか: `deepseek_hidden_state.csv`, `llama_hidden_state.csv`, `qwen_hidden_state.csv`

### メモリ不足エラー
**原因**: Hidden Stateデータが4096次元と高次元のため
**解決方法**: 
- Jupyter Notebookを再起動
- 他のプロセスを終了してメモリを確保

### パッケージ不足エラー
```bash
pip install --upgrade pandas numpy matplotlib seaborn scikit-learn
```

## ⚠️ 注意事項

- **このツールはデータ解析専用**です。Hidden State抽出は`../hidden_state_extraction/`で行います
- CSVファイルの読み込みに時間がかかる場合があります（高次元データのため）
- 解析結果の解釈については[SLIDES.md](../SLIDES.md)を参照してください

## 📋 期待される実行結果

正常に実行されると以下の結果が得られます：
- **DEEPSEEK**: HBDI ≈ 0.79（断定的）
- **LLAMA**: HBDI ≈ 0.03（極めて慎重） 
- **QWEN**: HBDI ≈ 0.02（極めて慎重）

**詳細な解釈と意味については[SLIDES.md](../SLIDES.md)のスライド12-13を参照してください。**
