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
| 完了フェーズ | Phase 1〜7 / 9 |
| テスト | 19 tests (Phase 6: 6, Phase 7: 6, Model: 7) |

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
│  └─ Phase 7: 分散AI基盤   │
│      ├─ 3D並列 (DP/TP/PP) │
│      ├─ FSDP/DeepSpeed    │
│      ├─ BF16/FP16/FP8     │
│      └─ NCCL通信+分散CKPT  │
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
python3 tests/test_model.py
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
| Phase 8 | Self Improving AI (自律改善、Online LoRA、経験学習) | 📋 |
| Phase 9 | AI Research Scientist (論文読解、実験自動化、新アルゴリズム提案) | 📋 |

📖 全体構想: [`VISION.md`](VISION.md) · [`PROJECT_PLAN.md`](PROJECT_PLAN.md) · [`ARCHITECTURE.md`](ARCHITECTURE.md)

---

## 🤝 協力者募集 — 計算資源が足りません！

TinyLLM は**ソフトウェアアーキテクチャとして完成**しています。
しかし、**実際にモデルを訓練するための GPU 計算資源が圧倒的に不足**しています。

### 必要なもの

| リソース | 用途 | 目安 |
|---------|------|------|
| **GPU クラスタ** | モデル訓練・蒸留 | A100 8〜64 枚級 × 2〜4 週間 |
| **大規模データセット** | 事前学習・指示チューニング | 1T+ トークンの高品質コード/対話データ |
| **クラウドクレジット** | AWS/GCP/Azure での実験 | $5,000〜$50,000 相当 |
| **人的リソース** | C 最適化, モデル蒸留, テスト | コア貢献者 |

### 協力方法

- 💻 **コード**: PR 大歓迎！ [`CONTRIBUTING.md`](CONTRIBUTING.md) (準備中)
- 🖥️ **GPU 提供**: 遊休 GPU があれば訓練に活用させてください
- 💰 **スポンサー**: [GitHub Sponsors](https://github.com/sponsors/RyoOtani) で支援
- 📊 **データ提供**: 高品質コード/対話データの寄贈
- 🧪 **テスト**: 様々な環境での動作検証

### 連絡先

- GitHub Issues: [Issue を作成](https://github.com/RyoOtani/DS4SmallestAIprjct/issues)
- メール: (準備中)
- Twitter/X: (準備中)

> **「単一バイナリの哲学」と「最先端 AI」の融合に、あなたの力を貸してください。**

## ライセンス

MIT License — 自由に使って、改良して、世界を変えてください。
