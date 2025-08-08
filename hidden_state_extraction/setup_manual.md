# `extract_hidden_states.py` 実行環境構築マニュアル（修正版）

## 概要

このスクリプトは、3つの大規模言語モデル（DeepSeek、Llama、Qwen）から隠れ状態ベクトルを抽出し、CSV形式で保存します。授業で学んだ「LLMのHidden Stateによる態度測定」を実際に体験できます。

**注意：このスクリプトの実行は任意です。技術的難易度が高いため、無理に実行する必要はありません。**

## 1. システム要件

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

## 2. Python環境のセットアップ

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

## 3. 仮想環境の作成（強く推奨）

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

## 4. 必要ライブラリのインストール

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

## 5. Hugging Faceアカウントの設定（重要）

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


## 6. スクリプトの実行

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
Saved: deepseek_jump_vectors.csv
========== Starting model: llama ==========
...
```

## 8. 出力ファイル

実行完了後、以下のCSVファイルが生成されます：
- `deepseek_jump_vectors.csv` - DeepSeekモデルの隠れ状態ベクトル
- `llama_jump_vectors.csv` - Llamaモデルの隠れ状態ベクトル  
- `qwen_jump_vectors.csv` - Qwenモデルの隠れ状態ベクトル

各ファイルの構造：
- **label列**: 質問ラベル（Q1_A_sensitive_F等）
- **d1〜d4096列**: 隠れ状態ベクトルの各次元の値

## 9. トラブルシューティング

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
- GPU ドライバーの更新

---

**重要**：このスクリプトは高い技術的要求を持つため、実行は任意です。実行できない場合でも授業の理解には影響ありません。