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

もしあなたが A100/H100 クラスタや豊富なクラウドクレジットを持っているなら、
**モデルを訓練して、GGUF ファイルをコミュニティに共有してください**。

```bash
# 1. モデル訓練 (あなたの GPU で)
torchrun --nproc_per_node=8 -m agent.phase7.cli train --config medium --data /your/dataset

# 2. GGUF エクスポート
python3 -c "
from model import create_model, export_model_to_gguf
model = create_model('medium')  # 訓練済みモデルをロード
export_model_to_gguf(model, 'tinyllm-medium.gguf', {...})
"

# 3. 共有 (Git LFS / Hugging Face / 直接 PR)
git lfs track "*.gguf"
git add tinyllm-medium.gguf
git commit -m "Add trained tinyllm-medium GGUF"
git push
```

### 欲しいモデル

| 優先度 | モデル | 活性化パラメータ | 必要 GPU | 用途 |
|--------|--------|----------------|---------|------|
| 🔴 最優先 | `small` | 3.0B | A100×4, 1週間 | 個人 PC で快適動作 |
| 🟠 高 | `medium` | 14.5B | A100×8, 2週間 | 本格ローカル開発 |
| 🟡 中 | `xlarge` | 43.9B | H100×16, 3週間 | プロフェッショナル用途 |
| 🟢 低 | `mega` / `giga` | 178B+ | H100×64, 4週間 | 研究・頂点性能 |

### 共有方法

- 📦 **Git LFS**: このリポジトリに直接 PR (100MB 以下の量子化モデル)
- 🤗 **Hugging Face**: `RyoOtani/tinyllm-models` にアップロード → Issue でお知らせください
- ☁️ **任意のストレージ**: Google Drive, Dropbox 等 → Issue でリンクを共有

### 訓練データについて

コード・対話データも同時に募集中です。良いデータがあれば良いモデルが生まれます。

> **「コードは書いた。あとは世界の GPU パワーでモデルを生み出すだけ。」**

## ライセンス

MIT License — 自由に使って、改良して、世界を変えてください。
