# Hidden State Extraction

このディレクトリには、LLMからHidden Stateを抽出するためのスクリプトと、既に抽出済みのデータが含まれています。

## 📂 このフォルダの目的

複数のLLM（DeepSeek-R1、LLaMA-3、Qwen3）に対立命題を提示し、最初の3語生成時のHidden Stateベクトル（4096次元）を抽出・保存します。

**理論的背景については[SLIDES.md](../SLIDES.md)を参照してください。**

## 📁 ファイル構成

```
hidden_state_extraction/
├── extract_hidden_states.py      # Hidden State抽出メインスクリプト
├── README.md                     # このファイル
└── results/                      # 抽出済みデータ
    ├── deepseek_hidden_state.csv    # DeepSeek-R1の抽出結果
    ├── llama_hidden_state.csv       # LLaMA-3の抽出結果
    └── qwen_hidden_state.csv        # Qwen3の抽出結果
```

## 📊 既存の抽出済みデータについて

`results/`フォルダには、既にHidden State抽出が完了したCSVファイルが含まれています：

### データ概要
- **対象モデル**: DeepSeek-R1-Distill-Llama-8B、Meta-Llama-3-8B-Instruct、Qwen3-8B
- **質問数**: 10組の対立命題ペア（政治的質問1組 + 自然科学質問9組）
- **抽出対象**: 各質問への回答の最初の3語生成時のHidden Stateの平均
- **データ形式**: 各モデル20行×4097列（label列 + 4096次元ベクトル）

### CSVファイル構造
```
label,d1,d2,d3,...,d4096
Q1_A_sensitive_F,-0.123,0.456,...,0.789
Q1_B_safe_F,0.234,-0.567,...,-0.123
...
```

**これらのデータは`../hidden_state_analysis/`で解析に使用されます。**

## ⚠️ 重要な注意事項

- **データ解析の学習目的では、既存のCSVファイルを使用してください**
- **新たなHidden State抽出は高度な技術的要件が必要です**
- **実行は完全に任意であり、学習には必須ではありません**

---

# 【上級者向け】extract_hidden_states.py の実行方法

以下は、新たにHidden State抽出を行いたい上級者向けの詳細な実行環境構築マニュアルです。

## システム要件

### 最小要件
- **OS**: Windows 10/11、macOS 10.15以上、Linux
- **Python**: 3.8以上（3.9以上推奨）
- **RAM**: 32GB以上**必須**（64GB推奨）
- **ストレージ**: 50GB以上の空き容量
- **インターネット接続**: 初回実行時にモデルファイルをダウンロード（約30-40GB）

### GPU使用の場合（大幅な高速化）
- **NVIDIA GPU**: 12GB VRAM以上（RTX 4070以上推奨）
- **CUDA Toolkit**: 11.8または12.4
- **RAM**: 64GB以上推奨

### 実行時間の目安
- **CPU環境（32GB RAM）**: 3-6時間程度
- **GPU環境（RTX 4090等）**: 20-40分程度
- **GPU環境（RTX 4070等）**: 40-80分程度

**⚠️ 重要：RAM 16GB以下のPCでは実行できません**

## Python環境のセットアップ

### Windows

1. **Python公式サイト**からPython 3.9以上をダウンロード
   - https://www.python.org/downloads/
   - インストール時に「Add Python to PATH」にチェック

2. **コマンドプロンプト**または**PowerShell**を開く

3. Pythonのバージョン確認
   ```cmd
   python --version
   ```

### macOS

1. **Homebrewのインストール**（未インストールの場合）
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Pythonのインストール**
   ```bash
   brew install python
   ```

3. バージョン確認
   ```bash
   python3 --version
   ```

## 仮想環境の作成（強く推奨）

### Windows
```cmd
# 作業ディレクトリを作成
mkdir hidden_states_project
cd hidden_states_project

# 仮想環境作成
python -m venv hidden_states_env

# 仮想環境の有効化
hidden_states_env\Scripts\activate

# 有効化の確認（プロンプトに(hidden_states_env)が表示される）
```

### macOS/Linux
```bash
# 作業ディレクトリを作成
mkdir hidden_states_project
cd hidden_states_project

# 仮想環境作成
python3 -m venv hidden_states_env

# 仮想環境の有効化
source hidden_states_env/bin/activate

# 有効化の確認（プロンプトに(hidden_states_env)が表示される）
```

## 必要ライブラリのインストール

### ステップ1: 基本ライブラリのインストール
```bash
# Hugging Face関連（認証に必要）
pip install huggingface_hub

# 基本ライブラリ
pip install transformers numpy pandas accelerate safetensors sentencepiece
```

### ステップ2: PyTorchのインストール

#### オプション1: CPU版（軽量、動作が遅い）
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### オプション2: GPU版（NVIDIA GPU必要、高速）

**CUDA 11.8の場合:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.4の場合:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### GPU環境の確認
```python
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA devices:', torch.cuda.device_count())"
```

## Hugging Faceアカウントの設定（重要）

### アカウント作成
1. https://huggingface.co/ でアカウントを作成
2. 無料アカウントで問題ありません

### 認証の実行
```bash
huggingface-cli login
```
- プロンプトでHugging Faceのトークンを入力
- トークンは https://huggingface.co/settings/tokens で作成

### Llamaモデルへのアクセス申請
1. https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct にアクセス
2. 「Request access」をクリック
3. 承認まで数時間～数日かかる場合があります

## スクリプトの実行

### 実行前の準備
- **他のアプリケーションを終了**（メモリ節約のため）
- **安定したインターネット接続を確保**（初回モデルダウンロードのため）
- **電源アダプターに接続**（ノートPCの場合）

### 実行コマンド
```bash
python extract_hidden_states.py
```

### 実行中の表示例
```
========== Starting model: deepseek ==========
[deepseek] Processing Q1_A_sensitive...
[deepseek] Processing Q1_B_safe...
...
Saved: deepseek_hidden_state.csv
========== Starting model: llama ==========
...
```

## 出力ファイル

実行完了後、以下のCSVファイルが`results/`フォルダに生成されます：
- `deepseek_hidden_state.csv` - DeepSeekモデルのHidden State
- `llama_hidden_state.csv` - LlamaモデルのHidden State
- `qwen_hidden_state.csv` - QwenモデルのHidden State

各ファイルの構造：
- **label列**: 質問ラベル（Q1_A_sensitive_F等）
- **d1〜d4096列**: Hidden Stateベクトルの各次元の値

## トラブルシューティング

### よくある問題と解決法

#### 1. メモリ不足エラー（最頻出）
```
CUDA out of memory / OutOfMemoryError
```
**解決法:**
- **全てのアプリケーションを終了**
- **ブラウザを完全に終了**
- **PCを再起動してから実行**
- CPU版PyTorchに変更

#### 2. Hugging Face認証エラー
```
401 Unauthorized / Access denied
```
**解決法:**
- `huggingface-cli login`を再実行
- Llamaモデルのアクセス申請を確認
- 代替モデルに変更

#### 3. CUDA関連エラー
```
CUDA initialization error
```
**解決法:**
- CPU版PyTorchに変更
- CUDA Toolkitのバージョンを確認
- GPUドライバーの更新

---

**重要**：このスクリプトは高い技術的要求を持つため、実行は任意です。実行できない場合でも学習の理解には影響ありません。既存のCSVファイルを使用して`../hidden_state_analysis/`で解析を進めてください。
