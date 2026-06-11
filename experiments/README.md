# 検証実験 ― 再現手順とスクリプト対応表

このディレクトリは、教材 [`../LECTURE_hidden_state.md`](../LECTURE_hidden_state.md) が示す主張を**実際に追試・再現**するためのコードと結果サマリです。すべて、複数のLocal LLM（Qwen3-4B / Mistral-7B / Phi-4-mini / DeepSeek-R1-Distill）を手元で動かして測定したものです。

## このディレクトリの構成

```
experiments/
├── README.md              # このファイル
├── requirements.txt       # 必要パッケージ（検証時の実バージョン）
├── questions_v1.yaml      # 評価用の質問36問（科学・正/偽・議論・政治 各9）
├── anchors.yaml           # 規格化用アンカー対立命題10組
├── PLAN.md                # 実験計画1（プロービング/一致率の設計）
├── PLAN2_dispersion.md    # 実験計画2（分布の広がり＝一貫性の設計）
├── scripts/               # 測定スクリプト（下表）
└── results/<model>/       # 各モデルの結果サマリ(JSON) + toorPIA入力CSV(1点)
```

## 動かし方

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt              # torch は環境のCUDAに合わせる
# 例: Qwen3-4B を 4bit で一貫性スコアを算出
python scripts/08_consistency_metric.py --model Qwen/Qwen3-4B --load-4bit
```

- **必要環境**: GPU（8GB VRAMで4bit量子化により8Bクラスまで可）。DeepSeek-R1系は `--no-think` を付ける（思考ブロックを空にプリフィル）。
- **GPUが無い場合**: `results/<model>/*.json` に測定済みの数値があるので、それを読んで集計・作図の部分だけ再現できる。`results/Qwen3-4B/dispersion_vectors.csv` は toorPIA 可視化（スライド8の図）の入力データ。

## スクリプト → 教材の主張・図 の対応

| スクリプト | 何を測るか | 教材での対応 |
|---|---|---|
| `logit_lens.py` | 末尾状態を語彙に射影し「次に何を言うか」を読む（層ごと） | スライド6（次の語を持つ／一点読みの限界） |
| `02_agreement.py` | 読み取った方向と実際の回答行動の一致率 | スライド6（読み取りが行動と一致） |
| `05_decision_point_check.py` | 同一質問内/文脈差での状態の決定性・ゆらぎの出所 | スライド4（鎖・ゆらぎの出所）, スライド5（文脈の中の意味） |
| `03_dispersion.py` | n試行の状態の広がり＋対立ペア距離の有意性検定 | スライド7（散らばり）, 旧HBDIの距離の有意性 |
| `06_window_spread.py` | 出だし何語まで見るかで広がりがどう変わるか（スライド8の入力 `first3_vectors.csv` もここで生成） | スライド7（出だし数語で測る理由） |
| `04_toorpia_map.py` | 状態群をtoorPIAベースマップで2D可視化 | スライド8（toorPIAマップ。入力は `results/Qwen3-4B/first3_vectors.csv`。完全同一行が多いため微小ジッタを加えて投入する） |
| `07_model_profile.py` | カテゴリ別の出だし広がりプロファイル | スライド11（配備マップの素地） |
| `08_consistency_metric.py` | アンカー規格化＋d′安全弁による一貫性スコア | スライド10-11（一貫性品質評価・物差し・安全弁） |
| `09_gen_full.py` + `10_validate.py` | 出だしの広がり vs 意味埋め込みの広がりの相関 | スライド12（検証(a)：読まない一貫性の妥当性） |

## 教材の図の再生成（scripts/figures/）

教材 `LECTURE_hidden_state.md` の各図は、以下のスクリプトで `../images/` に再生成できます。いずれも **GPU不要**（`results/` の収録データから描画。fig03 は座標キャッシュ `toorpia_first3_xy.npy` があればオフラインで動作し、無ければ toorPIA API に投入して生成）。

| スクリプト | 生成する図 | 入力データ |
|---|---|---|
| `figures/fig01_determinism.py` | `images/01_determinism.png`（スライド4・鎖の模式図） | なし（純粋な模式図） |
| `figures/fig02_spread_two_questions.py` | `images/02_spread_two_questions.png`（スライド7・2問の点群） | `full_first3.npz`, `full_texts.json` |
| `figures/fig03_toorpia_map.py` | `images/03_toorpia_map.png`（スライド8・toorPIAマップ） | `first3_vectors.csv`, `toorpia_first3_xy.npy` |
| `figures/fig04_consistency_standard.py` | `images/04_consistency_standard.png`（スライド12・3パネル） | 各モデルの `consistency.json`, `profile.json` |
| `figures/fig06_validation.py` | `images/06_validation.png`（スライド13・検証散布図） | `validation_a.json` |

※ `latent_attitude_consensus_module.png`（スライド14の構成図）はスクリプト生成ではなく、元PDF「LLM (SLM) WorkerのHidden State可視化による出力の最適化イメージ.pdf」を `pdftoppm -png -r 150` で変換したものです。

## 主要な数値（results/ に保存済み）

- `consistency.json`: D_ref（意味的距離の単位）, d′（物差しの妥当性）, 規格化inconsistency（全体/カテゴリ別）
- `profile.json`: カテゴリ別の出だし広がり, 事実正答率, 政治コミット指数
- `validation_a.json`: hidden側の広がり vs 意味側の広がりの相関（Qwen ρ=0.54 / Mistral ρ=0.64）
- `agreement.json` / `logit_lens.json` / `dispersion.json` / `window_spread.json` / `toorpia_spread.json`: 各実験の生数値

注：`results/` には小さなJSONサマリと、toorPIA可視化の入力CSV 2点（`first3_vectors.csv`＝出だし3トークン版・講義スライド8の図の入力、`dispersion_vectors.csv`＝回答全体平均版・PLAN2の実験用）を収録。その他の大きな中間ベクトル（`*.npz` 等）はスクリプト再実行で再生成できるため同梱していない。
