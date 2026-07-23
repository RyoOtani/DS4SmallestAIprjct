# 🚀 TinyLLM かんたん学習ガイド

**GPU を持っているあなたへ — この手順だけで TinyLLM モデルを訓練し、世界と共有できます。**

かかる時間: **セットアップ 5 分 + 訓練 1〜4 週間（GPU 性能による）**

---

## 前提条件

- Linux (Ubuntu 20.04+ 推奨) / WSL2
- NVIDIA GPU (A100/H100 推奨, RTX 4090 でも可)
- CUDA 12.1+ / Python 3.10+
- ディスク空き容量 500GB+ (データセット + チェックポイント用)

---

## Step 1: リポジトリをクローン

```bash
git clone https://github.com/RyoOtani/DS4SmallestAIprjct.git
cd DS4SmallestAIprjct
pip install -r requirements.txt
```

---

## Step 2: データを準備

### 方法 A: サンプルデータですぐ試す (数分)

```bash
# ダミーデータ生成 (動作確認用)
python3 -c "
import torch
torch.randint(0, 32000, (100000,)).numpy().astype('int32').tofile('data/train.bin')
torch.randint(0, 32000, (10000,)).numpy().astype('int32').tofile('data/val.bin')
"
mkdir -p data
```

### 方法 B: 本格データセット (推奨)

```bash
# The Stack (コード), FineWeb (Web), または任意の .jsonl / .bin ファイルを data/ に配置
# 対応フォーマット:
#   - .bin: int32 のトークン列 (Raw binary)
#   - .jsonl: {"tokens": [...]} または {"text": "..."} 形式
```

---

## Step 3: GPU 構成に合わせてモデルサイズを選ぶ

```bash
# 利用可能なモデル一覧を表示
python3 -c "from model import list_models; [print(f'  {m[\"name\"]:20s} {m[\"total_params\"]:>10s} total  {m[\"active_params\"]:>10s} active') for m in list_models()]"
```

### GPU 枚数別おすすめ

| あなたの GPU | おすすめモデル | コマンド |
|-------------|--------------|---------|
| RTX 4090 ×1 | `nano` (1.5B) | `--config nano` |
| RTX 4090 ×2 | `small` (14.8B) | `--config small` |
| A100 ×4 | `small` (14.8B) | `--config small` |
| A100 ×8 | `medium` (109B) | `--config medium` |
| H100 ×8 | `xlarge` (369B) | `--config xlarge` |
| H100 ×16+ | `xxlarge` (1.3T) | `--config xxlarge` |

---

## Step 4: 訓練スタート！

### シングル GPU (一番簡単)

```bash
python3 -m agent.phase7.cli train \
  --config nano \
  --data data/train.bin \
  --batch-size 2 \
  --lr 3e-4 \
  --max-steps 100000 \
  --output-dir checkpoints
```

### マルチ GPU (torchrun)

```bash
# 8 GPU の場合
torchrun --nproc_per_node=8 -m agent.phase7.cli train \
  --config small \
  --data data/train.bin \
  --batch-size 2 \
  --grad-accum 4 \
  --lr 2e-4 \
  --max-steps 200000 \
  --output-dir checkpoints
```

### カスタム並列度 (TP/PP 指定)

```bash
# 例: 8 GPU で TP=2, PP=2, DP=2
torchrun --nproc_per_node=8 -m agent.phase7.cli train \
  --config medium \
  --tp 2 --pp 2 \
  --data data/train.bin \
  --max-steps 300000
```

### DeepSpeed ZeRO-3 (メモリ節約)

```bash
torchrun --nproc_per_node=8 -m agent.phase7.cli train \
  --config small \
  --deepspeed \
  --data data/train.bin
```

---

## Step 5: GGUF にエクスポート

```bash
# 訓練完了後、GGUF 形式に変換 (tinyllm C ランタイムで動かせるように)
python3 -c "
from model import create_model, export_model_to_gguf

model = create_model('small')
# 注意: 実際は訓練済みチェックポイントをロードしてください
# checkpoint = torch.load('checkpoints/checkpoint_final/training_state.pt')
# model.load_state_dict(checkpoint['model'])

config_dict = {
    'name': 'tinyllm-small', 'hidden_dim': 2048, 'n_layers': 32,
    'n_heads': 32, 'n_kv_heads': 8, 'vocab_size': 65536,
    'max_seq_len': 8192, 'use_moe': True, 'use_mla': True,
    'kv_latent_dim': 512, 'n_experts': 64, 'n_active_experts': 6,
    'tie_word_embeddings': False,
}

export_model_to_gguf(model, 'tinyllm-small-q4.gguf', config_dict, use_q4_0=True)
print('✅ GGUF exported! Run: ./tinyllm run tinyllm-small-q4.gguf')
"
```

---

## Step 6: 世界と共有 🌍

### A. GitHub に直接 PR

```bash
git lfs install
git lfs track "*.gguf"
git add tinyllm-small-q4.gguf
git commit -m "Add trained TinyLLM-small GGUF model"
git push origin main
# → Pull Request を作成してください！
```

### B. Hugging Face にアップロード

```bash
pip install huggingface_hub
huggingface-cli upload RyoOtani/tinyllm-models tinyllm-small-q4.gguf .
# → Issue で「アップロードしたよ」とお知らせください
```

### C. Google Drive / Dropbox 等

リンクを [Issue](https://github.com/RyoOtani/DS4SmallestAIprjct/issues) に貼ってください。

---

## 🔧 トラブルシューティング

### CUDA Out of Memory

```bash
# バッチサイズを小さく
--batch-size 1 --grad-accum 8

# DeepSpeed ZeRO-3 を有効化
--deepspeed

# より小さいモデルに
--config nano
```

### 訓練が遅い

```bash
# 混合精度を有効に (デフォルトで ON)
# torch.compile で高速化
python3 -c "import torch; print(torch.__version__)"  # 2.0+ 推奨

# データを事前トークナイズ
python3 -c "
from model.training.data import TextDataset
ds = TextDataset('data/train.bin', seq_len=8192)
print(f'{len(ds)} samples ready')
"
```

### チェックポイントから再開

```bash
# 途中で止まっても大丈夫
torchrun --nproc_per_node=8 -m agent.phase7.cli train \
  --config small \
  --output-dir checkpoints  # 自動的に最新チェックポイントから再開
```

---

## 📊 訓練の進捗確認

```bash
# 別ターミナルで
watch -n 5 nvidia-smi

# ログ確認
tail -f checkpoints/checkpoint_*/training_state.pt  # バイナリなので代わりに
ls -la checkpoints/  # チェックポイントが増えていれば順調
```

---

## ✅ チェックリスト

- [ ] リポジトリをクローンした
- [ ] `pip install -r requirements.txt` が通った
- [ ] データを準備した
- [ ] GPU に合ったモデルサイズを選んだ
- [ ] `torchrun` で訓練を開始した
- [ ] 訓練が完了した (またはチェックポイントが保存された)
- [ ] GGUF にエクスポートした
- [ ] GitHub / Hugging Face で共有した 🎉

---

## 🤝 よくある質問

**Q: 訓練にどれくらい時間がかかる？**  
A: `nano` なら RTX 4090 で約 2〜3 日。`small` なら A100×4 で約 1 週間。`medium` 以上はクラスタ推奨。

**Q: 途中で止まったら？**  
A: チェックポイントから自動再開されます。同じコマンドをもう一度実行してください。

**Q: データは何を使えばいい？**  
A: まずサンプルで動作確認 → The Stack (コード) + FineWeb (Web) の組み合わせがおすすめ。

**Q: 訓練済みモデルはどこに置けばいい？**  
A: GitHub LFS で PR、または Hugging Face にアップロードして Issue でお知らせください。

---

> **あなたの GPU が、次の TinyLLM を生み出します。**
