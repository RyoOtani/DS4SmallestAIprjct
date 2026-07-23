# TinyLLM — A lightweight C-based LLM inference and AI runtime — ds4 Personal AI

> 「インターネット不要、月額課金なし。個人の PC で GPT‑5 に匹敵する開発生産性を発揮し、かつ自ら成長し続ける AI パートナー」

**tinyllm** は antirez のミニマリズム (ds4) と DeepSeek の効率性 (MoE/MLA/MTP) を融合した、C 言語約 5,000〜10,000 行の単一バイナリ AI システムです。

```
$ tinyllm run model.gguf          # CLI 対話モード
$ tinyllm serve model.gguf 8420   # HTTP API サーバー
$ tinyllm agent model.gguf "バグを修正して"  # 自律エージェント
```

## 特徴

| 項目 | 値 |
|------|-----|
| バイナリサイズ | ~200 KB (C コードのみ) |
| メモリ使用量 | 2〜8 GB (モデル込み) |
| 最小モデル | 活性化 600M〜2.4B パラメータ |
| 推奨モデル | 活性化 7B MoE (全パラ 30B) |
| 外部依存 | ゼロ (同梱不要) |
| 設定ファイル | 不要 (起動時自動スキャン) |
| 量子化 | GGUF Q4_0 / Q6_K 混合 |
| コンテキスト長 | 最大 8192 トークン |

## アーキテクチャ

```
+-----------------------+
| tinyllm (C 単一バイナリ) |
| ├─ モデルローダー (GGUF)  |
| ├─ MoE ルーター          |
| ├─ MLA アテンション       |
| ├─ トークナイザー (BPE)    |
| ├─ サンプラー             |
| ├─ エージェントループ      |
| ├─ HTTP サーバー          |
| ├─ RAG ベクトル検索       |
| └─ 長期記憶ストア         |
+-----------------------+
         │ パイプ / サブプロセス
         ↓
+-----------------------+
| 外部ツール群            |
| ├─ tree-sitter (AST)   |
| ├─ codeindex (依存グラフ)|
| ├─ gcc/pytest (テスト)   |
| ├─ ベクトル検索 (RAG)    |
| └─ Docker/podman (sandbox)|
+-----------------------+
```

## 主要アルゴリズム

- **MLA** (Multi-head Latent Attention): KV キャッシュを低ランク潜在空間に圧縮。メモリ使用量を 1/8 に削減
- **MoE** (Mixture of Experts): トークンごとにトップ 2 エキスパートのみ計算。活性化パラメータを大幅削減
- **MTP** (Multi-Token Prediction): 次の N トークンを同時予測。知識密度を向上
- **SwiGLU** FFN: ゲーテッド活性化関数による高品質な非線形変換

## ビルド

```bash
# 必要なもの: C11 コンパイラ + POSIX 環境
make

# デバッグビルド
make debug

# インストール
make install    # → /usr/local/bin/tinyllm
```

## 使い方

### 1. モデルの準備

```bash
# 蒸留で小さなモデルを作る
python python/train/distill.py \
  --teacher deepseek-ai/DeepSeek-V3 \
  --student Qwen/Qwen3-5-9B \
  --data data/code_train.jsonl \
  --output output/student.gguf

# GRPO でコード生成能力を強化
python python/train/grpo_train.py \
  --model output/student_model \
  --problems data/code_problems.jsonl \
  --output output/grpo_finetuned

# LoRA アダプターでユーザー適応
python python/train/lora_adapter.py \
  --model output/student_model \
  --feedback data/user_feedback.jsonl \
  --output output/lora_adapter.bin
```

### 2. CLI で対話

```bash
tinyllm run model.gguf

>>> Python で FizzBuzz を書いて
def fizzbuzz(n):
    for i in range(1, n+1):
        if i % 15 == 0: print("FizzBuzz")
        elif i % 3 == 0: print("Fizz")
        elif i % 5 == 0: print("Buzz")
        else: print(i)

[0.3s, 3.2 GB RSS]
```

### 3. HTTP API サーバー

```bash
tinyllm serve model.gguf 8420

# 別ターミナルで
curl -X POST http://localhost:8420/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "// C function to reverse a string\nchar* reverse(", "max_tokens": 256}'
```

### 4. 自律エージェント

```bash
tinyllm agent model.gguf "src/utils.c のバグを調査して修正し、テストが通ることを確認して"
```

## ディレクトリ構造

```
tinyllm/
├── include/
│   ├── config.h          # コンパイル時定数・プラットフォーム検出
│   └── tinyllm.h         # 全 API 宣言
├── src/
│   ├── main.c            # エントリーポイント
│   ├── model.c           # GGUF モデルローダー
│   ├── transformer.c     # Transformer 順伝播
│   ├── attention.c       # MLA + RMS Norm + RoPE
│   ├── moe.c             # MoE ルーティング + SwiGLU FFN
│   ├── quantize.c        # 量子化・逆量子化 (Q4_0, Q8_0)
│   ├── tokenizer.c       # BPE トークナイザー (FIM 対応)
│   ├── sampler.c         # Top-k / Top-p サンプリング
│   ├── inference.c       # 推論ループ・自己回帰生成
│   ├── agent.c           # 自己修正エージェントループ
│   ├── tools.c           # 外部ツール実行
│   ├── http.c            # 最小 HTTP サーバー
│   ├── rag.c             # RAG ベクトル検索
│   ├── memory.c          # 長期記憶ストア
│   └── util.c            # メモリ管理・I/O・ハッシュ
├── python/
│   ├── train/
│   │   ├── distill.py    # 知識蒸留 (teacher → student)
│   │   ├── grpo_train.py # GRPO 強化学習
│   │   └── lora_adapter.py # LoRA 適応学習
│   └── tools/
│       ├── code_analyzer.py # AST 解析・依存グラフ
│       └── sandbox.py       # サンドボックス実行
├── Makefile
├── CMakeLists.txt
├── requirements.txt
└── README.md
```

## フェーズ別ロードマップ

| フェーズ | ドキュメント | 目標 |
|---------|-------------|------|
| Phase 2 | [`PHASE2.md`](PHASE2.md) | 知識注入、蒸留、指示チューニング |
| Phase 3 | [`PHASE3_MULTI_AGENT.md`](PHASE3_MULTI_AGENT.md) | マルチエージェント協調 |
| Phase 4 | [`PHASE4_AI_SOFTWARE_ENGINEER.md`](PHASE4_AI_SOFTWARE_ENGINEER.md) | AI ソフトウェアエンジニア |
| Phase 5 | [`PHASE5_AUTONOMOUS_CODING.md`](PHASE5_AUTONOMOUS_CODING.md) | 完全自律コーディング |

## ライセンス

MIT License
