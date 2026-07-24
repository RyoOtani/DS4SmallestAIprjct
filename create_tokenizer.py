#!/usr/bin/env python3
"""
Creates a standalone TinyLLM tokenizer — zero external downloads needed.

This script builds a BPE tokenizer from scratch using the `tokenizers` library.
The resulting tokenizer files are committed to the repo so that the notebook
can load them without any internet access.

Vocabulary size: 65,536
Special tokens: <s>, </s>, <pad>, <unk>, <fim_prefix>, <fim_suffix>, <fim_middle>

Usage:
    python create_tokenizer.py
    → Outputs tokenizer/ directory (tokenizer.json + config files)
"""

import json
import os
import sys

def create_tokenizer(output_dir="tokenizer"):
    """Create a standalone TinyLLM tokenizer."""
    os.makedirs(output_dir, exist_ok=True)

    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
        from tokenizers.normalizers import NFKC
        HAS_TOKENIZERS = True
    except ImportError:
        HAS_TOKENIZERS = False

    if HAS_TOKENIZERS:
        # ── Build BPE tokenizer from scratch ──────────────────────
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

        # Normalizer: Unicode NFKC
        tokenizer.normalizer = NFKC()

        # Pre-tokenizer: ByteLevel
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # Decoder
        tokenizer.decoder = decoders.ByteLevel()

        # Post-processor for BOS/EOS
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> <s> $B </s>",
            special_tokens=[
                ("<s>", 0),
                ("</s>", 1),
            ],
        )

        # Trainer: byte-level training with minimal data
        trainer = trainers.BpeTrainer(
            vocab_size=65536,
            special_tokens=[
                "<s>", "</s>", "<pad>", "<unk>",
                "<fim_prefix>", "<fim_suffix>", "<fim_middle>",
                "<tool_call>", "</tool_call>",
                "<scratchpad>", "</scratchpad>",
            ],
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        # Train on a minimal corpus (ASCII + code snippets)
        training_corpus = generate_training_corpus()
        tokenizer.train_from_iterator(training_corpus, trainer)

        # Save
        tokenizer.save(f"{output_dir}/tokenizer.json")
        print(f"✅ Saved {output_dir}/tokenizer.json")

        # ── HuggingFace-compatible config files ───────────────────
        config = {
            "add_prefix_space": False,
            "added_tokens_decoder": {
                "0": {"content": "<s>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
                "1": {"content": "</s>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
                "2": {"content": "<pad>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
                "3": {"content": "<unk>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
            },
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "model_max_length": 8192,
            "tokenizer_class": "PreTrainedTokenizerFast",
            "clean_up_tokenization_spaces": False,
        }
        with open(f"{output_dir}/tokenizer_config.json", 'w') as f:
            json.dump(config, f, indent=2)

        special_tokens_map = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "additional_special_tokens": [
                "<fim_prefix>", "<fim_suffix>", "<fim_middle>",
                "<tool_call>", "</tool_call>",
                "<scratchpad>", "</scratchpad>",
            ],
        }
        with open(f"{output_dir}/special_tokens_map.json", 'w') as f:
            json.dump(special_tokens_map, f, indent=2)

        print(f"✅ Saved {output_dir}/tokenizer_config.json")
        print(f"✅ Saved {output_dir}/special_tokens_map.json")
        total_size = sum(os.path.getsize(f"{output_dir}/{f}") for f in os.listdir(output_dir))
        print(f"\n📊 Tokenizer size: {total_size/1024/1024:.2f} MB")

        # Quick test
        test = "def hello_world():\n    print('Hello, TinyLLM!')\n"
        encoded = tokenizer.encode(test)
        print(f"📝 Test: '{test.strip()}' → {len(encoded.ids)} tokens")

    else:
        # ── Fallback: manual tokenizer files without tokenizers lib ──
        print("⚠️  `tokenizers` library not available. Creating minimal config only.")
        print("   Install with: pip install tokenizers")

        config = {
            "add_prefix_space": False,
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "model_max_length": 8192,
            "tokenizer_class": "PreTrainedTokenizerFast",
            "clean_up_tokenization_spaces": False,
        }
        with open(f"{output_dir}/tokenizer_config.json", 'w') as f:
            json.dump(config, f, indent=2)

        special_tokens_map = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        }
        with open(f"{output_dir}/special_tokens_map.json", 'w') as f:
            json.dump(special_tokens_map, f, indent=2)

        print(f"✅ Created minimal config files in {output_dir}/")
        print(f"⚠️  Run `pip install tokenizers` and re-run to build full tokenizer.")


def generate_training_corpus():
    """Generate bilingual (EN + JP) training data for the BPE tokenizer."""

    samples = [
        # ── English code ──────────────────────────────────────
        "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr)//2]",
        "class Node:\n    def __init__(self, val):\n        self.val = val\n        self.left = None",
        "import asyncio\nasync def fetch(url):\n    async with aiohttp.ClientSession() as s:",
        "#include <stdio.h>\nint main() {\n    printf(\"Hello, World!\\n\");\n    return 0;\n}",
        "function fibonacci(n) { if (n <= 1) return n; return fibonacci(n-1) + fibonacci(n-2); }",
        "fn main() { let v = vec![1, 2, 3]; for x in &v { println!(\"{}\", x); } }",
        "package main\nimport \"fmt\"\nfunc main() { fmt.Println(\"Hello\"); }",
        "interface User { id: number; name: string; email: string; }",

        # ── English general ────────────────────────────────────
        "The transformer architecture uses self-attention to process sequences.",
        "Gradient descent is an optimization algorithm that minimizes the loss function.",
        "A binary search tree maintains the invariant that left < root < right.",
        "HTTP/1.1 200 OK\nContent-Type: application/json\n\n{\"status\": \"ok\"}",
        "git commit -m \"Add feature\" && git push origin main",
        "SELECT * FROM users WHERE email LIKE '%@example.com' ORDER BY created_at DESC",

        # ── 日本語コード・技術文書 ────────────────────────────
        "def 線形探索(配列, 目標値):\n    for i, 値 in enumerate(配列):\n        if 値 == 目標値:\n            return i\n    return -1",
        "class 顧客管理:\n    def __init__(self, 名前, メール):\n        self.名前 = 名前\n        self.メール = メール",
        "import requests\nresponse = requests.get('https://api.example.com/データ')\nprint(response.json())",
        "#include <stdio.h>\nint main() {\n    printf(\"こんにちは世界\\n\");\n    return 0;\n}",

        # ── 日本語一般文書 ────────────────────────────────────
        "トランスフォーマーアーキテクチャは自己注意機構を用いて系列データを処理します。",
        "深層学習モデルの学習には大規模なデータセットと計算資源が必要です。",
        "このプロジェクトは個人のPCで動作する省メモリなAIアシスタントを目指しています。",
        "関数型プログラミングは副作用のない純粋関数を組み合わせてプログラムを構築します。",
        "データベースのインデックスはB-tree構造を用いて高速な検索を実現します。",
        "マイクロサービスアーキテクチャでは各サービスが独立してデプロイ可能です。",
        "セキュリティ対策として入力値のバリデーションとサニタイズが重要です。",
        "GitHubでプルリクエストを作成してコードレビューを依頼してください。",
        "エラーハンドリングにはtry-except文を使用して適切に例外を捕捉します。",
        "分散システムではCAP定理に基づいて一貫性と可用性のトレードオフを考慮します。",
        "テスト駆動開発では最初にテストを書いてから実装を行います。",
        "コンテナ技術を使うことで環境に依存しないアプリケーションの実行が可能です。",
        "機械学習モデルの評価には適合率と再現率のバランスが重要です。",
        "オブジェクト指向設計では単一責任の原則に従ってクラスを分割します。",
        "並行処理ではミューテックスやセマフォを使って排他制御を行います。",

        # ── 日本語コードコメント・ドキュメント ────────────────
        "# この関数は二分探索を実装しています\n# 計算量はO(log n)です\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
        "// ユーザー認証を行うミドルウェア\n// JWTトークンを検証してリクエストを処理します\nfunction authMiddleware(req, res, next) {\n    const token = req.headers.authorization;\n    if (!token) return res.status(401).json({ error: '認証が必要です' });\n    try {\n        const decoded = jwt.verify(token, process.env.JWT_SECRET);\n        req.user = decoded;\n        next();\n    } catch (err) {\n        return res.status(403).json({ error: 'トークンが無効です' });\n    }\n}",
        "<!-- ユーザープロフィール画面 -->\n<div class=\"profile\">\n    <h1>プロフィール</h1>\n    <p>名前: {{ user.name }}</p>\n    <p>メール: {{ user.email }}</p>\n</div>",
        "# 設定ファイル\n# データベース接続情報\ndatabase:\n  host: localhost\n  port: 5432\n  name: myapp\n  user: admin",

        # ── 日本語技術ブログ・ドキュメント調 ──────────────────
        "この記事ではPythonでの非同期処理の実装方法について解説します。asyncioライブラリを使用することで、I/O待ちの間に他の処理を実行できます。",
        "Rustの所有権システムはメモリ安全性をコンパイル時に保証する革新的な仕組みです。 borrow checkerが参照のライフタイムを追跡します。",
        "Dockerを使用することで開発環境の構築が大幅に簡略化されます。 Dockerfileに必要な依存関係を記述するだけで再現可能な環境が作成できます。",
        "APIの設計においてはバージョニング戦略が重要です。 URLパスにバージョンを含める方法とヘッダーで指定する方法があります。",
        "GitHub Actionsを使ったCI/CDパイプラインの構築方法を説明します。 テストの自動実行から本番環境へのデプロイまでを自動化できます。",
        "Reactの状態管理にはuseStateとuseReducerの2つのフックがあります。 複雑な状態遷移にはuseReducerが適しています。",

        # ── 英日混在 ──────────────────────────────────────────
        "TensorFlowは Googleが開発した深層学習フレームワークで、PythonとC++のAPIを提供しています。",
        "Kubernetes はコンテナオーケストレーションツールで、ポッドの自動スケーリング機能があります。",
        "TypeScript は JavaScriptに型システムを追加した言語で、大規模開発に適しています。",
        "PostgreSQL はオープンソースのリレーショナルデータベースで、JSONデータ型もサポートしています。",
        "VSCode はマイクロソフトが開発したコードエディタで、拡張機能が豊富です。",
    ]

    # Duplicate and shuffle to create enough data for BPE training
    import itertools
    for i, sample in enumerate(itertools.cycle(samples)):
        if i >= 5000:
            break
        yield sample


if __name__ == '__main__':
    create_tokenizer()
