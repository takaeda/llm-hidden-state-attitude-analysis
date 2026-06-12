# 検証実験 ― 再現手順とスクリプト対応表

このディレクトリは、教材（本編 [`../LECTURE_hidden_state.md`](../LECTURE_hidden_state.md)・応用資料 [`../ADVANCED.md`](../ADVANCED.md)）が示す主張を**実際に追試・再現**するためのコードと結果サマリです。すべて、複数のLocal LLM（Qwen3-4B / Mistral-7B / Phi-4-mini / DeepSeek-R1-Distill）を手元で動かして測定したものです。

## このディレクトリの構成

```
experiments/
├── README.md              # このファイル
├── requirements.txt       # 必要パッケージ（検証時の実バージョン）
├── questions_v1.yaml      # 評価用の質問36問（科学・正/偽・議論・政治 各9）
├── anchors.yaml           # 規格化用アンカー対立命題10組
├── PLATFORM_NOTES.md      # macOS / Windows での実行方法（概略）
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

- **必要環境**: Linux + GPU（8GB VRAMで4bit量子化により8Bクラスまで可）。DeepSeek-R1系は `--no-think` を付ける（思考ブロックを空にプリフィル）。
- **macOS / Windows で動かす場合**: [`PLATFORM_NOTES.md`](PLATFORM_NOTES.md) を参照。
- **GPUが無い場合**: `results/<model>/*.json` に測定済みの数値があるので、それを読んで集計・作図の部分だけ再現できる。`results/Qwen3-4B/dispersion_vectors.csv` は toorPIA 可視化（スライド8の図）の入力データ。
- **toorPIA を使う場合**（`04_toorpia_map.py` や toorPIA 系の図の新規生成）: 別途 API 利用の手続きが必要です。講師に問い合わせてください。

## スクリプト → 教材の主張・図 の対応

※下表の「スライドN」は **ADVANCED.md（応用資料）** のスライド番号。

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
| `09_gen_full.py` + `10_validate.py` | 出だしの広がり vs 意味埋め込みの広がりの相関 | スライド13（検証(a)：読まない一貫性の妥当性） |
| `11_tree_expansion.py` | 浅い木展開（厳密確率）vs サンプリング頻度の一致検証 | スライド14（木展開Q&A・浅い領域） |
| `12_complex_task.py` | 複雑課題での木の枝数増殖と、フル回答の意味クラスタ | スライド14（木展開Q&A・壁1と壁2） |
| `13_layer_sweep.py` | 立場・意味の分離度と言い回しノイズの層プロファイル | スライド17（開かれた問い・予備実験） |
| `14_pooling_compare.py` | プーリング方式（出だし3語/全文平均/文末トークン × 最終層/中間層）の比較 | 検証スライドの精密化（出だし3語の妥当性） |

## 教材の図の再生成（scripts/figures/）

教材の各図は、以下のスクリプトで `../images/` に再生成できます（`fig_e*` ＝本編用の平易図、`fig0*` ＝ADVANCED用）。いずれも **GPU不要**（`results/` の収録データから描画。toorPIA系の図は座標キャッシュ（`toorpia_*_xy.npy`）があればオフラインで動作し、無ければ toorPIA API に投入して生成）。**2次元化はすべて toorPIA**（fig09 のみ比較のため PCA も併用）。

| スクリプト | 生成する図 | 入力データ |
|---|---|---|
| `figures/fig01_determinism.py` | `images/01_determinism.png`（スライド4・鎖の模式図） | なし（純粋な模式図） |
| `figures/fig02_spread_two_questions.py` | `images/02_spread_two_questions.png`（スライド7・2問の点群） | `full_first3.npz`, `full_texts.json`, `toorpia_pair_first3_xy.npy` |
| `figures/fig03_toorpia_map.py` | `images/03_toorpia_map.png`（ADVANCED 8・toorPIAマップ） | `full_first3.npz`, `toorpia_map12_full_nonorm_xy.npy` |
| `figures/fig04_consistency_standard.py` | `images/04_consistency_standard.png`（スライド12・3パネル） | 各モデルの `consistency.json`, `profile.json` |
| `figures/fig06_validation.py` | `images/06_validation.png`（スライド13・検証散布図） | `validation_a.json` |
| `figures/fig07_tree_limit.py` | `images/07_tree_limit.png`（スライド14・木展開の限界） | `complex_task.json`, `complex_task_vectors.npz`, `toorpia_sales_xy.npy` |
| `figures/fig08_layer_sweep.py` | `images/08_layer_sweep.png`（スライド17・層スイープ予備実験） | `layer_sweep.json` |
| `figures/fig_e01_chain.py` | `images/e01_chain.png`（本編3・サイコロの鎖） | なし（模式図） |
| `figures/fig_e02_two_questions.py` | `images/e02_two_questions.png`（本編5・2問の点群） | `full_first3.npz`, `full_texts.json`, `toorpia_pair_first3_xy.npy` |
| `figures/fig09_pca_vs_toorpia.py` | `images/09_pca_vs_toorpia.png`（ADVANCED 8・PCAとの比較） | 上記の点群データ + toorPIA座標キャッシュ2点 |
| `figures/fig_e03_map.py` | `images/e03_map.png`（本編6・頭の中の地図） | `full_first3.npz`, `toorpia_map12_full_nonorm_xy.npy` |
| `figures/fig_e04_stability.py` | `images/e04_stability.png`（本編8・安定性マップ） | 各モデルの `consistency.json` |

※ `latent_attitude_consensus_module.png`（本編9 / ADVANCEDスライド15の構成図）はスクリプト生成ではなく、元PDF「LLM (SLM) WorkerのHidden State可視化による出力の最適化イメージ.pdf」を `pdftoppm -png -r 150` で変換したものです。

## 主要な数値（results/ に保存済み）

- `consistency.json`: D_ref（意味的距離の単位）, d′（物差しの妥当性）, 規格化inconsistency（全体/カテゴリ別）
- `profile.json`: カテゴリ別の出だし広がり, 事実正答率, 政治コミット指数
- `validation_a.json`: hidden側の広がり vs 意味側の広がりの相関（Qwen ρ=0.54 / Mistral ρ=0.64）
- `pooling_compare.json`: プーリング方式比較（出だし3語が最良、文末トークンは打ち切り回答では不成立）
- `agreement.json` / `logit_lens.json` / `dispersion.json` / `window_spread.json` / `toorpia_spread.json`: 各実験の生数値

注：`results/` には結果サマリ(JSON)に加え、教材図の再生成に必要なデータ（toorPIA入力CSV 2点、`full_first3.npz`、`complex_task_vectors.npz`、座標キャッシュ）を収録。これら以外の中間生成物はスクリプト再実行で再生成できるため同梱していない。
