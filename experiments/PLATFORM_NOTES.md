# macOS / Windows での実行方法（概略）

スクリプト群は **Linux + NVIDIA GPU（CUDA）** で開発・検証しています。他OSで動かす場合の要点をまとめます。

## まず知っておくこと（OS共通）

- **GPUが要るのは測定スクリプト（`scripts/*.py`）だけ**です。図の再生成（`scripts/figures/`）と結果の集計は `results/` の収録データから動くため、**どのOSでもGPU不要**で実行できます。
- 環境差が出るポイントは次の3つです。
  1. **bitsandbytes（4bit量子化）**: CUDA前提のライブラリ。macOSでは使えません。
  2. **PyTorchのデバイス**: Linux/Windows は CUDA、macOS は MPS（Apple Silicon GPU）。
  3. **日本語フォント**: 図スクリプトは `IPAexGothic` を指定。OS標準フォントへの変更が必要。

## macOS（Apple Silicon）

```bash
python3 -m venv .venv && source .venv/bin/activate
# bitsandbytes を除いてインストール（CUDA専用のため）
grep -v bitsandbytes requirements.txt > requirements-mac.txt
pip install -r requirements-mac.txt   # torch は標準wheelでMPS対応
```

- **4bit量子化は使えない**ので、`--load-4bit` を**付けずに**実行します（bf16でロードされ、`device_map="auto"` がMPSに載せます）。
  ```bash
  python scripts/08_consistency_metric.py --model Qwen/Qwen3-4B
  ```
- **メモリ**: bf16ロードのため、4Bクラスで約8GB、7–8Bクラスで約16GBの統合メモリをモデルが占有します。7–8Bクラスは**32GB以上のマシン推奨**。
- **`05_decision_point_check.py` と `06_window_spread.py` は4bitロードがコードに直書き**されています。macOSで動かすには、`BitsAndBytesConfig` の部分を他スクリプトの非4bit側と同様の `torch_dtype=torch.bfloat16, device_map="auto"` のロードに書き換えてください。
- bfloat16 で MPS のエラーが出る場合は `torch.float16`（または `float32`）に変更します。
- **フォント**: 図スクリプトの `plt.rcParams["font.family"] = "IPAexGothic"` を `"Hiragino Sans"` に変更。

## Windows

### 推奨: WSL2（Ubuntu）

NVIDIA GPU があれば **WSL2 上の Ubuntu で Linux の手順がそのまま使えます**（bitsandbytes含む）。これが最も確実です。

1. Windows側に最新の NVIDIA ドライバをインストール（WSL内へのドライバ導入は不要）。
2. `wsl --install -d Ubuntu` で Ubuntu を導入。
3. WSL内で README どおりに venv 作成 → `pip install -r requirements.txt`（torch は CUDA 12.x ビルドを選択）。
4. 日本語フォントも Linux と同じ: `sudo apt install fonts-ipaexfont`

### ネイティブWindowsで動かす場合

- venv の有効化はコマンドが異なります: `.venv\Scripts\Activate.ps1`（PowerShell）。
- torch は CUDA ビルドの wheel（`--index-url https://download.pytorch.org/whl/cu121` など）を指定してインストール。
- bitsandbytes は近年のバージョンで Windows ネイティブの wheel が提供されていますが、**本教材では未検証**です。動かない場合は `--load-4bit` を外し bf16 で実行してください（VRAM 16GB以上推奨）。
- **フォント**: `IPAexGothic` を `"Yu Gothic"` または `"Meiryo"` に変更。

## GPUが無い／うまく動かないときの逃げ道

`results/<model>/*.json` に測定済みの数値をすべて収録しています。測定をスキップして、集計・作図（`scripts/figures/`、GPU不要・全OS対応）だけ再現する形でも教材の図はすべて再生成できます。詳細は [README.md](README.md) を参照してください。
