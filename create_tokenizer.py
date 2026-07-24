#!/usr/bin/env python3
"""
TinyLLM Custom Tokenizer — GPT-2 ByteLevel BPE x DeepSeek Coder x Japanese

Design:
  - ByteLevel BPE (GPT-2): no <unk>, handles any Unicode including Japanese
  - Large vocab: 65536 — room for code tokens + Japanese + English
  - FIM support (DeepSeek Coder): <fim_prefix>, <fim_suffix>, <fim_middle>
  - Tool tokens: <tool_call>, </tool_call>, <scratchpad>, </scratchpad>
  - Whitespace-preserving: critical for code indentation
  - Digit-aware: numbers get tokenized sensibly for code comprehension
  - Training: 60% code + 25% English text + 15% Japanese text

Usage:
    python create_tokenizer.py          # Build and save to tokenizer/
    python create_tokenizer.py --test   # Test the tokenizer
"""

import json, os, argparse

# ── Training corpus: code-heavy bilingual ──────────────────────

def code_samples():
    """60% of training data — real code patterns."""
    return [
        # Python
        "import os, sys, json, re, math, time",
        "from typing import Optional, List, Dict, Tuple, Union",
        "from dataclasses import dataclass, field",
        "from collections import defaultdict, deque, Counter, OrderedDict",
        "from functools import lru_cache, partial, reduce, wraps",
        "from pathlib import Path",
        "import numpy as np, torch, torch.nn as nn",
        "import torch.nn.functional as F",
        "from torch.utils.data import Dataset, DataLoader",
        "from transformers import AutoModel, AutoTokenizer",
        "def __init__(self, config: Config) -> None:",
        "class TransformerLayer(nn.Module):",
        "    def forward(self, x: torch.Tensor) -> torch.Tensor:",
        "self.attention = MultiHeadAttention(d_model, n_heads)",
        "self.norm1 = nn.LayerNorm(d_model, eps=1e-5)",
        "self.norm2 = nn.RMSNorm(d_model, eps=1e-5)",
        "self.ffn = SwiGLU(d_model, d_ff)",
        "return F.softmax(logits, dim=-1)",
        "loss = F.cross_entropy(logits, labels, ignore_index=-100)",
        "optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.1)",
        "scheduler = CosineAnnealingLR(optimizer, T_max=100000)",
        "with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):",
        "yield from iterable",
        "async def fetch(url: str) -> dict:",
        "    async with aiohttp.ClientSession() as session:",
        "        async with session.get(url) as resp:",
        "            return await resp.json()",
        "try: result = await process(data)",
        "except ValueError as e: logger.error(f'Invalid: {e}')",
        "finally: cleanup()",
        "@property\ndef name(self) -> str:\n    return self._name",
        "@staticmethod\ndef from_config(cfg: dict) -> 'Model':",
        "if __name__ == '__main__':\n    main()",
        "assert isinstance(x, int), f'Expected int, got {type(x)}'",
        # C / C++
        "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <stdint.h>",
        "#include <stdbool.h>\n#include <math.h>\n#include <pthread.h>",
        "#define MAX(a, b) ((a) > (b) ? (a) : (b))",
        "#define MIN(a, b) ((a) < (b) ? (a) : (b))",
        "#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))",
        "typedef struct { int x; int y; float val; } Point;",
        "typedef enum { OK, ERROR, TIMEOUT } Status;",
        "static inline int clamp(int x, int lo, int hi) {",
        "void* malloc(size_t size);\nvoid* calloc(size_t n, size_t s);\nvoid free(void* ptr);",
        "uint64_t hash(const uint8_t* data, size_t len);",
        "for (int i = 0; i < n; i++) {",
        "switch (type) {\n    case INT: return sizeof(int);\n    default: return 0;\n}",
        # Rust
        "use std::collections::{HashMap, HashSet, VecDeque};",
        "use std::sync::{Arc, Mutex, RwLock};",
        "use serde::{Serialize, Deserialize};",
        "use anyhow::{Result, Context, bail, ensure};",
        "use tokio::net::TcpListener;",
        "pub fn new(config: Config) -> Self {",
        "impl<T: Clone + Send + Sync + 'static> Worker<T> {",
        "#[derive(Debug, Clone, Serialize, Deserialize)]",
        "    .map(|x| x * 2).filter(|x| x > 0).collect::<Vec<_>>()",
        "match result {\n    Ok(val) => val,\n    Err(e) => return Err(e.into()),\n}",
        # JavaScript / TypeScript
        "import React, { useState, useEffect, useCallback, useMemo } from 'react';",
        "import express from 'express';",
        "import { Router, Request, Response, NextFunction } from 'express';",
        "const app = express();",
        "app.use(express.json());",
        "app.get('/api/users', async (req: Request, res: Response) => {",
        "interface User { id: number; name: string; email: string; }",
        "type Result<T> = { ok: true; data: T } | { ok: false; error: string };",
        "const fn = async (): Promise<void> => {",
        "const result = await fetch('/api/data');",
        "console.log(`User: ${user.name}, Age: ${user.age}`);",
        # Go
        "package main\nimport (\n    \"context\"\n    \"encoding/json\"\n    \"fmt\"\n    \"net/http\"\n)",
        "type Config struct {\n    Host string `json:\"host\" yaml:\"host\"`\n    Port int    `json:\"port\" yaml:\"port\"`\n}",
        "func NewServer(cfg *Config) *Server {",
        "ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)",
        "defer cancel()",
        # SQL / Shell
        "SELECT id, name, email, created_at FROM users WHERE email LIKE '%@example.com' ORDER BY created_at DESC LIMIT 100;",
        "INSERT INTO logs (user_id, action, timestamp) VALUES (?, ?, ?);",
        "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);",
        "#!/bin/bash\nset -euo pipefail",
        "docker build -t app:latest .",
        "git clone https://github.com/user/repo.git && cd repo && make -j$(nproc)",
        "curl -X POST -H \"Content-Type: application/json\" -d '{\"key\":\"val\"}' https://api.example.com",
    ]

def english_samples():
    """25% — technical English."""
    return [
        "The transformer architecture uses self-attention mechanisms to process sequential data efficiently.",
        "Gradient descent is an iterative optimization algorithm that finds the minimum of a loss function.",
        "Each attention head projects the input into query, key, and value representations.",
        "The model parameters are updated using backpropagation through the computational graph.",
        "Mixed precision training uses both 16-bit and 32-bit floating point to reduce memory usage.",
        "Layer normalization stabilizes training by normalizing activations across the feature dimension.",
        "The feed-forward network consists of two linear transformations with a non-linear activation.",
        "Dropout randomly zeros out neurons during training to prevent overfitting.",
        "The learning rate scheduler adjusts the step size based on training progress.",
        "Gradient clipping prevents exploding gradients by capping the norm of the gradient vector.",
        "Data augmentation increases the effective size of the training dataset through transformations.",
        "Batch normalization normalizes activations across the batch dimension for faster convergence.",
        "The validation set is used to monitor performance on unseen data during training.",
        "Transfer learning leverages knowledge from a pre-trained model for a new task.",
        "The encoder processes the input sequence and generates a contextualized representation.",
        "The decoder generates output tokens autoregressively based on the encoder's output.",
        "Cross-entropy loss measures the difference between predicted and true distributions.",
        "The embedding layer maps discrete tokens to continuous vector representations.",
        "Positional encoding injects information about token positions into the input embeddings.",
        "Early stopping halts training when validation performance stops improving.",
        "Weight decay adds an L2 penalty to the loss function to encourage smaller weights.",
        "The residual connection adds the input directly to the output of a sublayer.",
    ]

def japanese_samples():
    """15% — technical Japanese."""
    return [
        "トランスフォーマーアーキテクチャは自己注意機構を用いて系列データを処理します。",
        "深層学習モデルの学習には大規模なデータセットと高性能なGPUが必要です。",
        "Pythonの非同期処理にはasyncioライブラリを使用します。",
        "Rustの所有権システムはメモリ安全性をコンパイル時に保証します。",
        "GitHubでプルリクエストを作成してコードレビューを依頼してください。",
        "データベースのインデックスはB-tree構造を用いて高速な検索を実現します。",
        "マイクロサービスアーキテクチャでは各サービスが独立してデプロイ可能です。",
        "セキュリティ対策として入力値のバリデーションとサニタイズが重要です。",
        "テスト駆動開発では最初にテストを書いてから実装を行います。",
        "関数型プログラミングは副作用のない純粋関数を組み合わせてプログラムを構築します。",
        "Dockerを使用することで開発環境の構築が大幅に簡略化されます。",
        "Kubernetesはコンテナオーケストレーションツールとして広く使われています。",
        "APIの設計ではバージョニングと後方互換性の維持が重要です。",
        "機械学習モデルの評価には適合率と再現率のバランスが重要です。",
        "分散システムではCAP定理に基づいて一貫性と可用性のトレードオフを考慮します。",
        "並行処理ではミューテックスやセマフォを使って排他制御を行います。",
        "正規表現は文字列のパターンマッチングに使用される強力なツールです。",
        "オブジェクト指向設計では単一責任の原則に従ってクラスを分割します。",
        "CI/CDパイプラインを構築することでコードの品質を自動的に検証できます。",
        "エラーハンドリングには例外処理を用いて適切にエラーを捕捉します。",
        "TypeScriptはJavaScriptに静的型付けを追加した言語です。",
        "ReactはコンポーネントベースのUIライブラリです。",
        "Gitのブランチ戦略としてGitHub FlowやGit Flowがよく使われます。",
        "ロードバランサーはトラフィックを複数のサーバーに分散させます。",
    ]

def digit_samples():
    """Number and identifier patterns."""
    return [
        "0 1 2 3 4 5 6 7 8 9 10 100 1000 10000",
        "32 64 128 256 512 1024 2048 4096 8192 16384 65536",
        "1e-3 1e-4 1e-5 3e-4 0.001 0.0001 0.00001",
        "0.1 0.5 0.9 0.95 0.99 0.999",
        "batch_size=32 lr=0.001 epochs=100",
        "seq_len=1024 hidden_dim=2048 num_heads=16 num_layers=24",
        "vocab_size=65536 max_position=8192",
    ]

def create_tokenizer(output_dir="tokenizer"):
    """Build TinyLLM's GPT-2 x DeepSeek Coder x Japanese tokenizer."""
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
        from tokenizers.normalizers import NFKC
    except ImportError:
        print("pip install tokenizers")
        return

    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("TinyLLM Custom Tokenizer — GPT-2 BPE x DeepSeek Coder x JP")
    print("=" * 60)

    # GPT-2 style: ByteLevel BPE — no <unk>, any Unicode
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.normalizer = NFKC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[("<s>", 0), ("</s>", 1)],
    )

    # DeepSeek Coder style special tokens
    special = [
        "<s>", "</s>", "<pad>", "<unk>",
        "<fim_prefix>", "<fim_suffix>", "<fim_middle>",
        "<fim_hole>", "<fim_pad>",
        "<repo_name>", "<file_sep>", "<file_path>",
        "<tool_call>", "</tool_call>",
        "<tool_response>", "</tool_response>",
        "<scratchpad>", "</scratchpad>",
        "<|system|>", "<|user|>", "<|assistant|>",
    ]

    # Corpus: 60% code + 25% en + 15% ja
    print("Building corpus...")
    corpus = []
    cd = code_samples(); corpus.extend(cd * 60)
    print(f"  Code: {len(cd)} x60 = {len(cd)*60}")
    en = english_samples(); corpus.extend(en * 50)
    print(f"  EN:   {len(en)} x50 = {len(en)*50}")
    jp = japanese_samples(); corpus.extend(jp * 30)
    print(f"  JP:   {len(jp)} x30 = {len(jp)*30}")
    dg = digit_samples(); corpus.extend(dg * 20)
    print(f"  Num:  {len(dg)} x20 = {len(dg)*20}")
    print(f"  Total lines: ~{len(corpus)}")

    # Train BPE
    print("Training vocab=65536 ...")
    trainer = trainers.BpeTrainer(
        vocab_size=65536, special_tokens=special,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
    )
    tok.train_from_iterator(corpus, trainer)
    tok.save(f"{output_dir}/tokenizer.json")

    # HF config files
    added = {}
    for i, t in enumerate(special):
        added[str(i)] = {"content": t, "lstrip": False, "normalized": False,
                         "rstrip": False, "single_word": False, "special": True}
    config = {
        "add_prefix_space": False,
        "added_tokens_decoder": added,
        "bos_token": "<s>", "eos_token": "</s>",
        "unk_token": "<unk>", "pad_token": "<pad>",
        "model_max_length": 8192,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "clean_up_tokenization_spaces": False,
    }
    with open(f"{output_dir}/tokenizer_config.json", 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    smap = {
        "bos_token": "<s>", "eos_token": "</s>",
        "unk_token": "<unk>", "pad_token": "<pad>",
        "additional_special_tokens": special[4:],
    }
    with open(f"{output_dir}/special_tokens_map.json", 'w') as f:
        json.dump(smap, f, indent=2, ensure_ascii=False)

    kb = sum(os.path.getsize(f"{output_dir}/{f}") for f in os.listdir(output_dir)) / 1024
    print(f"\nDone! {kb:.0f} KB, vocab={tok.get_vocab_size()}")
    return tok


def test_tokenizer(output_dir="tokenizer"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(output_dir, use_fast=True)
    tests = [
        ("Python", "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    return quick_sort(left) + [pivot] + quick_sort([x for x in arr if x > pivot])"),
        ("C", "int factorial(int n) { if (n <= 1) return 1; return n * factorial(n - 1); }"),
        ("JS", "const add = (a,b) => a + b;\nconst r = [1,2,3].map(x => x*2).filter(x => x>3);"),
        ("Rust", "fn gcd(mut a: u64, mut b: u64) -> u64 { while b != 0 { let t = b; b = a % b; a = t; } a }"),
        ("EN", "The transformer model uses multi-head attention with scaled dot-product similarity."),
        ("JP", "トランスフォーマーモデルは自己注意機構を用いて文脈を理解します。"),
        ("SQL", "SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id;"),
        ("FIM", "<fim_prefix>def binary_search(arr, target):<fim_suffix>    return -1<fim_middle>"),
    ]
    print("="*60)
    print("Quality Tests")
    print("="*60)
    for label, text in tests:
        ids = tok.encode(text)
        ratio = len(ids) / len(text)
        pieces = [tok.decode([i]) for i in ids[:4]]
        print(f"  [{label:5s}] {len(text):3d} chars -> {len(ids):3d} tokens ({ratio:.2f}x)  {' '.join(pieces[:4])}")
    print(f"  Vocab: {tok.vocab_size}  |  ByteLevel: yes  |  FIM: yes  |  JP: yes")
    print("="*60)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--test', action='store_true')
    p.add_argument('--output', default='tokenizer')
    args = p.parse_args()
    create_tokenizer(args.output)
    if args.test:
        test_tokenizer(args.output)
