# AIRoA MoMa Value Function (201-bin)

`Hugging Face Transformers` を使って、AIRoA MoMa の 2視点画像 + 言語入力から分布型 value function を学習する実装です。

## 実装内容
- value 定義（`v_norm in [-1,0]`）と `201-bin` 離散化を仕様どおり実装
- `L_max(task)` を全episodeから算出して正規化に使用
- episode ごとに固定 `K=frames_per_episode` の timestep をサンプリング
- `t -> i` 写像: `i = round(t/(L-1)*(N_video-1))` を使用（動画区間timestampも考慮）
- 2視点入力:
  - `siglip2_gemma3_270m`: head/hand を別エンコードして融合
  - `paligemma2_3b_pt224`: head/hand を横結合モザイクして1枚入力
- `tune_mode`: `head | lora | full`
- `eval_vf.py`: val split で CE loss / continuous MAE
- `scripts/compute_epsilons.py`: task別 30 percentile epsilon を JSON 出力

## ディレクトリ構成

```text
repo_root/
  requirements.txt
  README.md
  train_vf.py
  eval_vf.py
  scripts/
    compute_epsilons.py
  src/
    data/
      airoa_moma_vf_dataset.py
    models/
      vf_backbones.py
      vf_model.py
    utils/
      video_io.py
      seed.py
      metrics.py
  tests/
```

## セットアップ

```bash
pip install -r requirements.txt
```

## 学習

```bash
python train_vf.py \
  --data_root /path/to/airoa-moma \
  --backbone siglip2_gemma3_270m \
  --tune_mode head \
  --max_train_steps 50 \
  --output_dir /tmp/vf_test
```

```bash
python train_vf.py \
  --data_root /path/to/airoa-moma \
  --backbone paligemma2_3b_pt224 \
  --tune_mode head \
  --max_train_steps 10 \
  --output_dir /tmp/vf_test2
```

## 評価

```bash
python eval_vf.py --checkpoint /tmp/vf_test --data_root /path/to/airoa-moma
```

## epsilon 計算

```bash
python scripts/compute_epsilons.py \
  --checkpoint /tmp/vf_test \
  --data_root /path/to/airoa-moma \
  --out eps.json
```

## 補足
- episode metadata は `episodes.jsonl` を優先して読みます。存在しない場合は `meta/episodes/*.parquet` を読みます。
- 動画デコードは `decord` を優先し、無ければ `torchvision.io.read_video` にフォールバックします。
- processor/tokenizer は dataset ではなく collator 側でバッチ処理します。
