# TinyLLM Tokenizer Specification

This document specifies the tokenizer used by TinyLLM models.

## Overview

| Property | Value |
|----------|-------|
| Algorithm | BPE (Byte-Pair Encoding) |
| Vocabulary size | 65,536 |
| Model type | `bpe` (HuggingFace: `tokenizers.models.BPE`) |
| Pre-tokenizer | ByteLevel with regex splitting |
| Bundled | `tokenizer/` (~45 KB, zero download) |
| 🇯🇵 nano/small 推奨 | `tokyotech-llm/Swallow-7b-v0.1` — 日本語トークン化効率◎ |
| Fallback | `Qwen/Qwen2.5-1.5B` (65,536 vocab, 同一アーキテクチャ) |

## Special Tokens

| Token | ID | Purpose |
|-------|----|---------|
| `<s>` | 0 | Beginning of sequence (BOS) |
| `</s>` | 1 | End of sequence (EOS) |
| `<pad>` | 2 | Padding |
| `<unk>` | 3 | Unknown token |
| Byte tokens | 4–259 | Byte-level fallback (256 tokens) |
| Regular vocab | 260–65510 | Standard BPE merges |
| `<fim_prefix>` | 65511 | FIM: prefix boundary |
| `<fim_suffix>` | 65512 | FIM: suffix boundary |
| `<fim_middle>` | 65513 | FIM: middle (generation point) |
| `<tool_call>` | 65514 | Tool call begin |
| `</tool_call>` | 65515 | Tool call end |
| `<scratchpad>` | 65516 | Scratchpad begin |
| `</scratchpad>` | 65517 | Scratchpad end |

## Loading

### ⚠️ なぜ Qwen のトークナイザーか？

TinyLLM は **まだ誰も大規模学習していない新プロジェクト** のため、現時点で独自の学習済みトークナイザーが存在しません。
`Qwen/Qwen2.5-1.5B` の BPE トークナイザーが TinyLLM と同一の語彙サイズ (65,536) であり、
アーキテクチャ互換性があるため **仮採用** しています。

学習済み TinyLLM モデルが完成したら、独自トークナイザーに置き換え予定です。

### Python (HuggingFace Transformers)

```python
from transformers import AutoTokenizer

# 初回: Qwen 互換トークナイザーをロード
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

# 自前トークナイザーとして保存 (次回から Qwen 不要)
tokenizer.save_pretrained("tinyllm-tokenizer")

# 次回以降: 自前トークナイザーを直接ロード
tokenizer = AutoTokenizer.from_pretrained("tinyllm-tokenizer")

# Add TinyLLM special tokens
tokenizer.add_special_tokens({
    'additional_special_tokens': [
        '<fim_prefix>', '<fim_suffix>', '<fim_middle>',
        '<tool_call>', '</tool_call>', '<scratchpad>', '</scratchpad>',
    ]
})

# Verify
assert len(tokenizer) == 65536  # after adding special tokens
```

### C (tinyllm native GGUF tokenizer)

The C runtime parses the tokenizer directly from the GGUF model file.
No external file is needed — the tokenizer data is embedded in the model.

## Pre-tokenization Rules

1. **Digits**: Groups of consecutive digits are kept together
2. **Letters**: Lowercase letters are merged with subsequent lowercase letters
3. **Unicode**: Multi-byte UTF-8 sequences are treated as individual bytes
4. **Whitespace**: Single spaces are kept; multiple spaces get a special token

## FIM (Fill-in-the-Middle) Format

For code infilling tasks, use the following format:

```
<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>
```

The model will generate the missing middle part.

## Tool Call Format

```
<tool_call>
<name>tool_name</name>
<params>{"key": "value"}</params>
</tool_call>
```

## Pre-tokenization for Training

For training, raw text should be pre-tokenized to `.bin` format:

```python
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

with open('data.txt') as f:
    text = f.read()

tokens = tokenizer.encode(text)
np.array(tokens, dtype=np.int32).tofile('data/train.bin')
```
