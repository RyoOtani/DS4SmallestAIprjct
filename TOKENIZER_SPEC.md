# TinyLLM Tokenizer Specification

This document specifies the tokenizer used by TinyLLM models.

## Overview

| Property | Value |
|----------|-------|
| Algorithm | BPE (Byte-Pair Encoding) |
| Vocabulary size | 65,536 |
| Model type | `bpe` (HuggingFace: `tokenizers.models.BPE`) |
| Pre-tokenizer | ByteLevel with regex splitting |
| Source | Compatible with `Qwen/Qwen2.5-1.5B` tokenizer |

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

### Python (HuggingFace Transformers)

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

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
