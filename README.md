# TinyLLM — ds4 Personal AI 〜 自律 AI ソフトウェアエンジニア基盤

> **「インターネット不要、月額課金なし。個人の PC で GPT‑5 に匹敵する開発生産性を発揮し、かつ自ら成長し続ける AI パートナー」**

**TinyLLM** は antirez のミニマリズム (ds4) と DeepSeek の効率性 (MoE/MLA/MTP) を融合した、
C 言語約 4,200 行の単一バイナリ推論エンジン + Python エージェント + 最先端モデルアーキテクチャの
**完全自律 AI ソフトウェアエンジニア基盤**です。

```bash
$ tinyllm run model.gguf                 # CLI 対話モード
$ tinyllm serve model.gguf 8420          # HTTP API サーバー
$ tinyllm agent model.gguf "バグを修正して" # 自律エージェント
$ torchrun --nproc_per_node=8 -m agent.phase7.cli train --config xlarge  # 分散学習
```

## 📊 プロジェクト規模

| 指標 | 値 |
|------|-----|
| 全ファイル数 | **~100 ファイル** |
| 総コード行数 | **~17,000 行** |
| C ランタイム | 4,208 行 (86 KB バイナリ) |
| Python エージェント | ~7,000 行 |
| AI モデル定義 | ~3,000 行 |
| モデルスケール | nano 1.5B 〜 giga 6.7T (9段階) |
| 完了フェーズ | Phase 1〜9 / 9 |
| テスト | 114 tests (Phase 6: 6, Phase 7: 6, Phase 8: 50, Phase 9: 45, Model: 7) |

## 特徴

| 項目 | 値 |
|------|-----|
| バイナリサイズ | 86 KB (C コードのみ) |
| モデル範囲 | 活性化 0.5B〜314B (総 1.5B〜6.7T) |
| 外部依存 | ゼロ (C ランタイム), pip install のみ (Python) |
| 量子化 | GGUF Q4_0 / Q6_K / Q8_0 混合 |
| 分散学習 | FSDP / DeepSpeed ZeRO-3 / TP/PP/DP/EP |
| 混合精度 | BF16 / FP16 / FP8 |
| 推論バックエンド | C (NEON/AVX2/Accelerate SIMD) |
| 設定 | 不要 (auto-detect) |

## アーキテクチャ

```
┌──────────────────────────┐     ┌───────────────────────────┐
│  tinyllm (C 単一バイナリ)  │     │  外部ツール群              │
│  ├─ GGUF ローダー (MoE)    │     │  ├─ tree-sitter (AST)     │
│  ├─ MLA アテンション       │────▶│  ├─ gcc/pytest (テスト)    │
│  ├─ MoE ルーター          │ パイプ│  ├─ ベクトル検索 (RAG)     │
│  ├─ BPE トークナイザー     │◀────│  ├─ Docker (サンドボックス) │
│  ├─ 投機的デコード         │     │  └─ mypy/pyright (型検査)  │
│  ├─ HTTP/CLI/デーモン      │     └───────────────────────────┘
│  ├─ RAG + 長期記憶        │
│  └─ 自己修正エージェント    │
└──────────────────────────┘
┌──────────────────────────┐
│  Cloud GPU Launcher       │
│  ├─ RunPod / VastAI       │
│  ├─ GPUSOROBAN / Lambda   │
│  └─ AWS / Azure / GCP     │
└──────────────────────────┘
┌──────────────────────────┐
│  Python エージェント群     │
│  ├─ Phase 3: マルチ協調   │
│  ├─ Phase 4: SW エンジニア│
│  ├─ Phase 5: 自律コーディング│
│  ├─ Phase 6: 深いコード理解│
│  │   ├─ 多言語AST/CallGraph│
│  │   ├─ アーキテクチャ分析 │
│  │   ├─ Quality Pipeline  │
│  │   ├─ Critic + Debugger │
│  │   └─ 構造化Tool Calling │
│  ├─ Phase 7: 分散AI基盤   │
│  │   ├─ 3D並列 (DP/TP/PP) │
│  │   ├─ FSDP/DeepSpeed    │
│  │   ├─ BF16/FP16/FP8     │
│  │   └─ NCCL通信+分散CKPT  │
│  └─ Phase 8: 自己改善AI   │
│      ├─ 自己評価+改善サイクル│
│      ├─ 経験再生+失敗DB    │
│      ├─ LoRAオンライン学習  │
│      ├─ メタ学習(戦略最適化)│
│      └─ 統合Orchestrator   │
│  └─ Phase 9: AI研究員     │
│      ├─ arXiv論文読解+解析 │
│      ├─ 実験設計+自動実行   │
│      ├─ 新アルゴリズム提案  │
│      ├─ 統計的仮説検証     │
│      └─ 研究レポート生成   │
└──────────────────────────┘
┌──────────────────────────┐
│  AI モデル (PyTorch)      │
│  ├─ 9スケール (1.5B〜6.7T)│
│  ├─ MLA + MoE + MTP       │
│  ├─ SwiGLU + RoPE(YaRN)   │
│  ├─ 訓練パイプライン       │
│  └─ GGUF Q4_0/Q6_K エクスポート│
└──────────────────────────┘
```

## 主要技術

- **MLA** (Multi-head Latent Attention): KV キャッシュを 1/8 に圧縮
- **MoE** (Mixture of Experts): 256+ エキスパート、top-8 のみ活性化
- **MTP** (Multi-Token Prediction): 最大 8 トークン同時予測
- **SwiGLU** FFN: ゲーテッド活性化、FP8 対応
- **RoPE + YaRN**: 32K コンテキスト対応の位置エンコーディング
- **投機的デコード**: N-gram draft + 一括検証
- **3D 並列**: データ/テンソル/パイプライン + エキスパート並列
- **アトミックパッチ**: バイナリ検出、事前検証、ロールバック
- **停止検出**: 同一エラー/診断のハッシュ+テキスト比較
- **mypy/pyright 連携**: 外部型チェッカーによる深い型推論
- **自律改善ループ**: 自己評価→診断→修正→測定の収束サイクル
- **経験再生**: 成功/失敗の経験を保存・検索し、類似タスクで再利用
- **障害データベース**: エラーパターンをハッシュ化して検索、過去の修正方法を提案
- **LoRA オンライン学習**: ベースモデル凍結・劣化時自動ロールバック・安全性ガードレール
- **メタ学習**: タスク横断パターン認識・戦略最適化・few-shot プロンプト生成
- **arXiv 論文読解**: 論文検索・構造化解析・知識グラフ構築・文献比較
- **実験自動設計**: A/Bテスト・グリッドサーチ・アブレーション研究の自動生成
- **アルゴリズム提案**: 既知技術の組み合わせによる新規アルゴリズム創出・新規性/実現性/インパクト評価
- **統計的仮説検証**: Welchのt検定・効果量(Cohen's d)・多重比較補正(Bonferroni/Holm)
- **研究レポート生成**: Markdown + LaTeX 形式の自動論文執筆
- **クラウドGPU抽象化**: RunPod/VastAI/GPUSOROBAN/Lambda/AWS/Azure/GCP/Local(Metal/CUDA) の統一インターフェース
- **Metal GPU バックエンド**: Apple M1/M2/M3/M4 GPU上のmatmul/flash-attention/MoE/RoPE/RMS Norm

## ビルド

```bash
# C ランタイム (C11 + POSIX)
make              # リリースビルド
make debug        # ASAN デバッグビルド
make install      # → /usr/local/bin/tinyllm

# Python (pip)
pip install -r requirements.txt
```

## クイックスタート

```bash
# モデル作成
python3 -c "from model import create_model; m = create_model('small'); print('OK')"

# コードレビュー
python3 -m agent.phase6.cli review src/main.c

# リポジトリ分析
python3 -m agent.phase6.cli scan .

# アーキテクチャ診断
python3 -m agent.phase6.cli analyze .

# 品質パイプライン
python3 -m agent.phase6.cli quality . --fix

# 分散学習 (8 GPU)
torchrun --nproc_per_node=8 -m agent.phase7.cli train --config xlarge

# テスト実行
python3 tests/test_phase6.py
python3 tests/test_phase7.py
python3 tests/test_phase8.py
python3 tests/test_phase9.py
python3 tests/test_model.py

# 自己改善AI
python3 -m agent.phase8.cli evaluate --test-command "python3 tests/"
python3 -m agent.phase8.cli replay --task-type code_generation
python3 -m agent.phase8.cli failures
python3 -m agent.phase8.cli skills
python3 -m agent.phase8.cli adapters
python3 -m agent.phase8.cli status

# AI研究員
python3 -m agent.phase9.cli search --query "mixture of experts transformer"
python3 -m agent.phase9.cli read --arxiv-id 1706.03762
python3 -m agent.phase9.cli propose --problem "efficient LLM inference" --techniques "MoE,MLA,distillation"
python3 -m agent.phase9.cli hypothesis test --control "0.85,0.86,0.84" --treatment "0.88,0.89,0.87"
python3 -m agent.phase9.cli research --topic "efficient transformer attention"
python3 -m agent.phase9.cli status

# クラウドGPU起動 (プロバイダー非依存)
python3 -m agent.cloud.launcher compare --config medium --gpus 8 --gpu-type a100-80gb
python3 -m agent.cloud.launcher launch --provider runpod --config xlarge
python3 -m agent.cloud.launcher prices --provider vastai
python3 -m agent.cloud.launcher status --provider aws

# ローカルGPU (Metal/CUDA)
python3 -m agent.cloud.launcher launch --provider local   # Auto-detect GPU
```

## フェーズ別ロードマップ

| フェーズ | 内容 | 状態 |
|---------|------|------|
| Phase 1 | TinyLLM Runtime (C 推論エンジン) | ✅ |
| Phase 2 | Repository Understanding (RAG, コード検索) | ✅ |
| Phase 3 | Multi-Agent Foundation (Planner, Coder, Memory) | ✅ |
| Phase 4 | AI Software Engineer (Provider, Tools, Planning) | ✅ |
| Phase 5 | Autonomous Coding Loop (Build→Test→Fix→Retry) | ✅ |
| Phase 6 | AI SW Engineer Professional (AST, Arch, Quality, Critic, Debug) | ✅ |
| Phase 7 | Distributed AI Platform (FSDP/DeepSpeed, TP/PP, FP8, NCCL) | ✅ |
| Phase 8 | Self Improving AI (自律改善、Online LoRA、経験学習) | ✅ |
| Phase 9 | AI Research Scientist (論文読解、実験自動化、新アルゴリズム提案) | ✅ |

📖 全体構想: [`VISION.md`](VISION.md) · [`PROJECT_PLAN.md`](PROJECT_PLAN.md) · [`ARCHITECTURE.md`](ARCHITECTURE.md)

---

## 🤝 学習済みモデルを共有してください！

TinyLLM の**ソフトウェアは完成**しています。コードはすべて揃っており、`python3 -m agent.phase7.cli train --config xlarge` を実行すれば、理論上どんな規模のモデルでも訓練できます。

**しかし、それを実行する GPU がありません。**

👉 **[🚀 かんたん学習ガイド (QUICKSTART_TRAIN.md)](QUICKSTART_TRAIN.md)** — コピペするだけで訓練スタート！

### あなたが GPU を持っているなら

## 🚀 1-Click Training Benchmark

> **Open in Colab or RunPod → Run all cells → Get a trained model in hours**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RyoOtani/DS4SmallestAIprjct/blob/main/TINYLLM_TRAIN_BENCHMARK.ipynb)
[![RunPod](https://img.shields.io/badge/RunPod-1--click-blue)](https://runpod.io/console/deploy?template=)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue)](https://kaggle.com/)

```bash
# Just run the notebook! No setup needed.
# Generates dummy data → Creates model → Trains → Exports GGUF
```

See [`TINYLLM_TRAIN_BENCHMARK.ipynb`](TINYLLM_TRAIN_BENCHMARK.ipynb) for the complete workflow.

---

## 🤝 Community & Contribution

### 🌟 Trained a model? Share it with the world!

**👉 https://huggingface.co/Ryo3desu/tinyllm-models**

This is the **official community repository** for community-trained TinyLLM models.  
Upload your `.gguf` or full weights and get **credited in the model card**!

```bash
huggingface-cli login
huggingface-cli upload Ryo3desu/tinyllm-models your-model.gguf tinyllm-small/your-name/
```

### 📋 How to contribute

1. **Train** — Use `TINYLLM_TRAIN_BENCHMARK.ipynb` on your GPU hardware
2. **Export** — Convert to GGUF format (Q4_0 recommended for sharing)
3. **Upload** — Push to our HF community repo or submit a GitHub PR
4. **Get credited** — Your name goes in the contributors hall of fame!

### 🎯 Priority Models

| Priority | Model | Active Params | GPU Required | Why |
|----------|-------|--------------|-------------|-----|
| 🔴 Highest | `small` | **3.0B** | A100×4, 1 week | Runs on any PC |
| 🟠 High | `medium` | 14.5B | A100×8, 2 weeks | Pro local dev |
| 🟡 Medium | `xlarge` | 43.9B | H100×16, 3 weeks | Professional |
| 🟢 Low | `mega` / `giga` | 178B+ | H100×64, 4 weeks | Research |

### 📦 Share Methods

| Method | How | Link |
|--------|-----|------|
| 🤗 **Hugging Face** | Upload to community repo | [Ryo3desu/tinyllm-models](https://huggingface.co/Ryo3desu/tinyllm-models) |
| 📦 **Git LFS** | Direct PR to this repo | [Submit PR](https://github.com/RyoOtani/DS4SmallestAIprjct/pulls) |
| ☁️ **Cloud storage** | GD/Dropbox → post link | [Open Issue](https://github.com/RyoOtani/DS4SmallestAIprjct/issues) |

---

## 🏗️ Design Decisions (設計上の意図)

Some components are intentionally limited in scope. This documents WHY.

### `export_gguf.py` — Export-only, not for inference

The `TinyLLMLayer.forward()` and `TinyLLMModel.forward()` in `export_gguf.py` raise `NotImplementedError`. **This is by design.**

- These classes exist solely to extract weight tensors for GGUF conversion.
- Actual inference runs in the **C runtime** (`./tinyllm run model.gguf`) or the notebook training model.
- Use `test_forward()` for basic structure verification during export.
- For Python inference: use the notebook's `TinyLLMModel` or HuggingFace `AutoModel`.

### Multi-Agent Roles — Template fallback without LLM

`agent/multi_agent/roles.py` roles (Planner, Coder, Tester, etc.) return **template-based responses** when no LLM provider is configured.

- With a provider: roles use LLM for intelligent reasoning.
- Without a provider: roles fall back to sensible defaults (inspect→design→implement→verify→review).
- This ensures the system **always functions**, even offline.

### Metal GPU — CPU fallback

`src/metal.m` functions return `0` (CPU fallback) when Metal is not available.

- If running on Apple Silicon WITH Metal: GPU-accelerated inference.
- If running on Intel Mac / Linux / without Metal: transparent CPU fallback.
- Auto-detected at compile time via `config.h` (`TL_HAS_METAL`).

### Provider Security — `shell=False` enforced

All `subprocess.run()` calls in provider code use `shell=False` with `shlex.split()`.

- Prevents command injection via user/prompt content.
- Verified by `tests/test_security.py` and CI pipeline.
- Regression: any new `shell=True` is caught by CI.

---

## 📚 Tokenizer & Dataset Specification

### Tokenizer

TinyLLM uses a **ByteLevel BPE tokenizer with 32,000 vocabulary** (v7.1), bundled in-repo:

| Property | Value |
|----------|-------|
| Type | ByteLevel BPE (GPT-2 architecture) + NFKC normalization |
| Vocab size | **32,000** (21 specials + 256 bytes + 31,723 BPE merges) |
| Special tokens | `<s>`=0, `</s>`=1, `<pad>`=2, `<unk>`=3, FIM (4-8), Tool (12-15), Chat (18-20) |
| Python (bundled) | `AutoTokenizer.from_pretrained('tokenizer')` — 3.3 MB, zero download |
| C runtime | `tl_tokenizer_load("tokenizer.tokbin")` — 950 KB binary |
| Rebuild | `python create_tokenizer.py` (any vocab size) |
| Verify | `python test_3way_tokenizer.py` — Python↔C↔TOKBIN consistency |

> **Note**: The default vocab size is 32,000 (not 65,536). For larger vocabularies,
> use `python create_tokenizer.py --vocab 65536`. See `TOKENIZER_SPEC.md` for full specification.

### Recommended Training Data

- **Primary**: [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2) — 900+ programming languages, 67TB
- **Code subset**: Python, C, C++, Rust, Go, JavaScript, TypeScript
- **Web + Code**: [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) — 15T tokens of web data
- **Pre-tokenized format**: Raw `.bin` files (int32 token IDs) or `.jsonl` with `{"tokens": [...]}`

### 蒸留用データ (Distillation Data)

別リポジトリで管理しています：

> **🤗 [DS4SmallestAIprjctDistilldata-DeepSeek-v4Flash-Pro-](https://github.com/RyoOtani/DS4SmallestAIprjctDistilldata-DeepSeek-v4Flash-Pro-)**

| File | Samples | Description |
|------|---------|-------------|
| `sample_code_distill.jsonl` | 200 | コード生成タスク (Python/C/Rust/Go/JS/TS) |
| `sample_instruction_distill.jsonl` | 100 | 技術概念の説明タスク |
| `sample_fim_distill.jsonl` | 61 | Fill-in-the-Middle コード補完 |
| `generate_large_dataset.py` | — | 50,000+ サンプル生成スクリプト |

```bash
# カスタムデータ生成 (DeepSeek API 不要)
python data/distillation/generate_large_dataset.py --output my_data.jsonl --n 1000

# 本番蒸留
python python/train/distill.py \
  --teacher deepseek-ai/DeepSeek-V3 \
  --student Qwen/Qwen3-5-9B \
  --data data/distillation/sample_code_distill.jsonl \
  --output output/distilled_model
```

```python
# Pre-tokenization example
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
tokens = tok.encode("Your training text here")
tokens.astype('int32').tofile('data/train.bin')
```

> **データも募集中！** 質の高いコード・対話データをお持ちでしたら、ぜひ共有してください。

---

## 🧪 Training your own model

```bash
# 1. Quick start (1-click)
#    → Open TINYLLM_TRAIN_BENCHMARK.ipynb in Colab

# 2. Single GPU
python3 -m agent.phase7.cli train \
  --config nano --data data/train.bin --batch-size 2 --max-steps 100000

# 3. Multi GPU
torchrun --nproc_per_node=8 -m agent.phase7.cli train \
  --config small --data data/train.bin --batch-size 2 --grad-accum 4 \
  --lr 2e-4 --max-steps 200000 --output-dir checkpoints

# 4. Export to GGUF
python3 -c "
from model.export.gguf_exporter import export_model_to_gguf
export_model_to_gguf(model, 'tinyllm-small-q4.gguf', config_dict, use_q4_0=True)
"

# 5. Run with C runtime
./tinyllm run tinyllm-small-q4.gguf
```

> **「コードは書いた。あとは世界の GPU パワーでモデルを生み出すだけ。」**
> **"I wrote the code. Now we just need the world's GPU power to train the models."**

## License

MIT License — Free to use, modify, and share.
