# TinyLLM — DS4 Personal AI 〜 自律 AI ソフトウェアエンジニア基盤

> **「インターネット不要、月額課金なし。個人の PC で AI 開発生産性を発揮し、かつ自ら成長し続ける AI パートナー」**

**TinyLLM** は C 言語の単一バイナリ推論エンジン + Python エージェント + Hierarchical MoE モデル + Web チャット UI の
**完全自律 AI ソフトウェアエンジニア基盤**です。

```bash
$ tinyllm run model.gguf                    # CLI 対話モード
$ tinyllm serve model.gguf 8420             # HTTP API サーバー
$ tinyllm agent model.gguf "バグを修正して"    # 自律エージェント
$ python agent/chat_server.py --port 8421   # Web チャット UI (ブラウザ)
$ python agent/chat_server.py --provider fallback --api-key sk-xxx  # 自動フォールバック
$ torchrun --nproc_per_node=8 train ...     # 分散学習
```

## 📊 プロジェクト規模

| 指標 | 値 |
|------|-----|
| 全ファイル数 | **~130 ファイル** |
| 総コード行数 | **~24,000 行** |
| C ランタイム | 4,200 行 (88 KB バイナリ) |
| Python エージェント | ~10,000 行 |
| Web チャット UI | シングルページ HTML/CSS/JS |
| AI モデル定義 | ~3,500 行 (Hierarchical MoE) |
| 本格ツール群 | 6 ツール (Git/Sandbox/Diff/Repair/Benchmark) |
| プロバイダー | 11 種類 (自前/DeepSeek/OpenAI/Claude/Gemini/...) |
| モデルスケール | small 530M 〜 Hierarchical MoE 14.5B |
| トークナイザー | nano 32K / small 72K (英日+Coding特化) |

## 特徴

| 項目 | 値 |
|------|-----|
| バイナリサイズ | 88 KB (C コードのみ) |
| 外部依存 | ゼロ (C ランタイム), pip install のみ (Python) |
| 量子化 | GGUF Q4_0 / Q6_K / Q8_0 混合 |
| 推論バックエンド | C (NEON/AVX2/Accelerate SIMD) |
| プロバイダー | 自前 TinyLLM + OpenAI/DeepSeek/Claude/Gemini/Groq/Ollama 他 |
| フォールバック | 🦾自前→🐋DeepSeek→🤖OpenAI→⚡Groq→🧩RuleBased (5段自動) |
| チャットUI | ブラウザベース、IME対応、コードコピー、モデル切替 |
| Web検索 | DuckDuckGo (ddgs), 24h キャッシュ |
| 設定 | 不要 (auto-detect) / APIキーはメモリのみ保持 |

## アーキテクチャ

```
┌──────────────────────────┐     ┌───────────────────────────┐
│  tinyllm (C 単一バイナリ)  │     │  Web チャット UI           │
│  ├─ GGUF ローダー          │     │  ├─ ブラウザベース UI      │
│  ├─ MLA アテンション       │     │  ├─ IME 対応 (日本語)      │
│  ├─ BPE トークナイザー     │     │  ├─ コードコピー機能       │
│  ├─ HTTP/CLI/デーモン      │     │  ├─ モデル切替 ⚙️          │
│  ├─ RAG + 長期記憶        │     │  └─ フォールバック状態表示   │
│  └─ 自己修正エージェント    │     └───────────────────────────┘
└──────────────────────────┘                 │
          │                                  │
┌─────────▼────────────────┐     ┌───────────▼───────────────┐
│  プロバイダー層            │     │  本格ツール群              │
│  ├─ TinyLLMProvider (自前) │     │  ├─ Git Checkpoint/Rollback│
│  ├─ OpenAICompatProvider   │     │  ├─ 安全サンドボックス     │
│  ├─ AnthropicProvider      │     │  ├─ 本格差分エディタ       │
│  ├─ GoogleProvider         │     │  ├─ 自律Self-Repair Loop  │
│  ├─ RuleBasedProvider      │     │  ├─ ベンチマーク基盤       │
│  └─ FallbackProvider (連鎖) │     │  └─ DuckDuckGo Web検索    │
└────────────────────────────┘     └───────────────────────────┘
┌────────────────────────────┐
│  Hierarchical MoE モデル    │
│  ├─ 2段階ルーティング       │
│  │   Domain Router (L1)    │
│  │   └→ Expert Router (L2) │
│  ├─ 4 ドメイン × 6 専門家   │
│  ├─ 共有エキスパート        │
│  ├─ 負荷分散 + 多様化       │
│  └─ GGUF エクスポート       │
└────────────────────────────┘
┌──────────────────────────┐
│  Python エージェント群     │
│  ├─ Phase 3: マルチ協調   │
│  ├─ Phase 4: SW エンジニア│
│  ├─ Phase 5: 自律コーディング│
│  ├─ Phase 6: 深いコード理解│
│  ├─ Phase 7: 分散AI基盤   │
│  ├─ Phase 8: 自己改善AI   │
│  └─ Phase 9: AI研究員     │
└──────────────────────────┘

## 主要技術

- **Hierarchical MoE** (階層型 Mixture of Experts): Domain Router (L1) → Expert Router (L2), 416 experts, 3.0B active / 14.5B total
- **MLA** (Multi-head Latent Attention): KV キャッシュを 1/8 に圧縮
- **SwiGLU** FFN: ゲーテッド活性化
- **RoPE**: 位置エンコーディング
- **プロバイダー抽象化**: BaseProvider (ABC) → 6種の実装 + 自動フォールバックチェーン
- **本格差分編集**: unified diff 生成・検証・ファジー適用 (git apply + patch)
- **安全サンドボックス**: CPU/メモリ/時間制限 + ネットワーク隔離
- **Git Checkpoint/Rollback**: コード変更の安全な実験・復元
- **自律Self-Repair Loop**: CHECKPOINT→DIAGNOSE→PLAN→EDIT→TEST→VERIFY
- **ベンチマーク基盤**: 5タスクのコード生成評価 + 回帰比較
- **英日+Coding特化トークナイザー**: BPE 72,000 語彙 (Python/Rust/TS/SQL/Shell)

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
# Web チャット UI 起動 (ブラウザで localhost:8421)
python agent/chat_server.py --port 8421

# 自動フォールバックで起動 (自前→DeepSeek→OpenAI→Groq→RuleBased)
python agent/chat_server.py --provider fallback --api-key sk-xxx

# モデル作成 (Hierarchical MoE)
python -c "from model.hierarchical_moe import create_hierarchical_moe_model; m = create_hierarchical_moe_model(dtype=torch.float16)"

# トークナイザー作成 (英日+Coding特化 72K)
python create_tokenizer_small.py --vocab 72000

# 本格ツール群
python -m agent.tools.git_checkpoint checkpoint "before-refactor"
python -m agent.tools.sandbox python "print(1+1)"
python -m agent.tools.benchmark run
python -m agent.tools.self_repair "TypeError at line 42"

# C ランタイム
make && ./tinyllm run model.gguf
./tinyllm serve model.gguf 8420
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

| Priority | Model | Active Params | Total Params | Architecture | GPU Required |
|----------|-------|--------------|-------------|-------------|-------------|
| 🔴 Highest | `hierarchical-moe` | **3.0B** | 14.5B | Hierarchical MoE (416 experts) | A100×4, 1 week |
| 🟠 High | `small` | 530M | 530M | Hierarchical MoE (192 experts, 576D) | A100×1, 3 days |
| 🟡 Medium | `medium` | 1.5B | 1.5B | Dense Transformer | A100×4, 1 week |
| 🟢 Low | `large` | 7B | 7B | Dense Transformer | A100×8, 2 weeks |

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

TinyLLM は 2 種類のトークナイザーを提供します：

| Property | nano (既存) | small (新) |
|----------|------------|------------|
| Type | ByteLevel BPE | ByteLevel BPE |
| Vocab size | **32,000** | **72,000** |
| 対象言語 | 汎用 | **英語 + 日本語 + コード特化** |
| 特殊トークン | 21 | **27** (FIM/Chat/Tool/言語マーカー) |
| 対応コード | — | Python/Rust/TypeScript/SQL/Shell |
| 作成 | `python create_tokenizer.py` | `python create_tokenizer_small.py` |
| Python | `AutoTokenizer.from_pretrained('tokenizer')` | `AutoTokenizer.from_pretrained('tokenizer_small')` |
| C runtime | `tokenizer.tokbin` (950 KB) | `tokenizer_small.tokbin` |

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

## 🧪 Training

```bash
# ── Hierarchical MoE モデル生成 ──
python -c "
from model.hierarchical_moe import create_hierarchical_moe_model
import torch
model = create_hierarchical_moe_model(dtype=torch.float16)  # ~30GB
print('Ready for training')
"

# ── トークナイザー作成 (英日+Coding 72K) ──
python create_tokenizer_small.py --vocab 72000

# ── 分散学習 (FSDP) ──
torchrun --nproc_per_node=8 -m training.trainer \
  --model hierarchical_moe --data data/train.bin \
  --batch-size 2 --grad-accum 4 --lr 2e-4 --max-steps 200000

# ── GGUF エクスポート ──
python export_tokenizer.py  # .tokbin for C runtime
```

> **「コードは書いた。あとは世界の GPU パワーでモデルを生み出すだけ。」**
> **"I wrote the code. Now we just need the world's GPU power to train the models."**

## License

MIT License — Free to use, modify, and share.
