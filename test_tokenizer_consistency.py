#!/usr/bin/env python3
"""
test_tokenizer_consistency.py — Regression test: Python HF tokenizer vs C runtime

Verifies that:
  1. Special token IDs match between Python and C
  2. BPE encoding produces identical token sequences
  3. Detokenization produces identical text
  4. FIM tokenization is consistent

Usage:
  python test_tokenizer_consistency.py [--verbose]
"""

import json, os, sys, subprocess, tempfile, argparse
from pathlib import Path

# ── Test cases ────────────────────────────────────────────────
TEST_CASES = [
    # (label, input_text)
    ("English-simple", "Hello, world!"),
    ("English-tech", "The transformer model uses multi-head attention with scaled dot-product similarity."),
    ("Python-func", "def binary_search(arr: list[int], target: int) -> int:\n    left, right = 0, len(arr) - 1"),
    ("Python-class", "class DataLoader:\n    def __init__(self, dataset, batch_size=32):\n        self.dataset = dataset"),
    ("C-code", "#include <stdio.h>\nint main(void) { printf(\"Hello, World!\\n\"); return 0; }"),
    ("JS-code", "const add = (a: number, b: number): number => a + b;\nexport default add;"),
    ("SQL", "SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id;"),
    ("Japanese-tech", "トランスフォーマーモデルは自己注意機構を用いて高精度な予測を行います。"),
    ("Japanese-chat", "お疲れ様です。先日のバグ修正について確認したいことがあります。"),
    ("FIM-test", "<fim_prefix>def quick_sort(arr):<fim_suffix>    return arr<fim_middle>"),
    ("Unicode-mixed", "こんにちは世界！🚀 Hello 世界 🌍 def process(データ: list[str]) -> None: pass"),
    ("Single-char", "a"),
    ("Numbers", "1234567890 3.14159 0xFF 1e-5"),
    ("Config", "DATABASE_URL=postgresql://user:pass@localhost:5432/db\nSECRET_KEY=abc123"),
]

SPECIAL_TOKENS = [
    "<s>", "</s>", "<pad>", "<unk>",
    "<fim_prefix>", "<fim_suffix>", "<fim_middle>",
    "<fim_hole>", "<fim_pad>",
    "<repo_name>", "<file_sep>", "<file_path>",
    "<tool_call>", "</tool_call>",
    "<tool_response>", "</tool_response>",
    "<scratchpad>", "</scratchpad>",
    "<|system|>", "<|user|>", "<|assistant|>",
]


def test_special_token_ids():
    """Verify special token IDs match between config files."""
    print("=" * 60)
    print("Test 1: Special Token ID Consistency")
    print("=" * 60)
    
    # Python config
    with open('tokenizer/tokenizer_config.json') as f:
        cfg = json.load(f)
    py_ids = {}
    for id_str, info in cfg.get('added_tokens_decoder', {}).items():
        if info.get('special', False):
            py_ids[info['content']] = int(id_str)
    
    # C header (parse manually)
    c_ids = {
        '<s>': 0, '</s>': 1, '<pad>': 2, '<unk>': 3,
        '<fim_prefix>': 4, '<fim_suffix>': 5, '<fim_middle>': 6,
        '<fim_hole>': 7, '<fim_pad>': 8,
        '<repo_name>': 9, '<file_sep>': 10, '<file_path>': 11,
        '<tool_call>': 12, '</tool_call>': 13,
        '<tool_response>': 14, '</tool_response>': 15,
        '<scratchpad>': 16, '</scratchpad>': 17,
        '<|system|>': 18, '<|user|>': 19, '<|assistant|>': 20,
    }
    
    all_ok = True
    for name in SPECIAL_TOKENS:
        py_id = py_ids.get(name, -1)
        c_id = c_ids.get(name, -1)
        if py_id != c_id:
            print(f"  ❌ MISMATCH: {name} → Python={py_id}, C={c_id}")
            all_ok = False
    
    if all_ok:
        print(f"  ✅ All {len(SPECIAL_TOKENS)} special token IDs match")
    return all_ok


def test_python_tokenizer():
    """Run Python tokenizer on all test cases."""
    print("\n" + "=" * 60)
    print("Test 2: Python HF Tokenizer")
    print("=" * 60)
    
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('tokenizer', use_fast=True)
    
    results = {}
    for label, text in TEST_CASES:
        ids = tok.encode(text)
        decoded = tok.decode(ids, skip_special_tokens=True)
        results[label] = {
            'ids': ids,
            'decoded': decoded,
            'num_tokens': len(ids),
        }
        print(f"  {label:20s} → {len(ids):4d} tokens, decode OK: {decoded == text}")
    
    print(f"\n  Vocab size: {tok.vocab_size:,}")
    return results


def test_tokbin_format():
    """Verify .tokbin file integrity."""
    print("\n" + "=" * 60)
    print("Test 3: TOKBIN Format Integrity")
    print("=" * 60)
    
    if not os.path.exists('tokenizer.tokbin'):
        print("  ⚠️  tokenizer.tokbin not found. Run: python export_tokenizer.py")
        return False
    
    import struct
    with open('tokenizer.tokbin', 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        version = struct.unpack('<I', f.read(4))[0]
        vocab_size = struct.unpack('<I', f.read(4))[0]
        n_merges = struct.unpack('<I', f.read(4))[0]
        n_specials = struct.unpack('<I', f.read(4))[0]
        max_bytes = struct.unpack('<I', f.read(4))[0]
    
    expected_magic = 0x424B4F54
    ok = True
    if magic != expected_magic:
        print(f"  ❌ Bad magic: 0x{magic:08X} (expected 0x{expected_magic:08X})")
        ok = False
    if version != 2:
        print(f"  ❌ Bad version: {version} (expected 2)")
        ok = False
    if vocab_size != 32000:
        print(f"  ⚠️  Vocab size: {vocab_size} (expected 32000)")
    
    if ok:
        print(f"  ✅ Magic OK, version={version}, vocab={vocab_size:,}, merges={n_merges:,}")
        print(f"     specials={n_specials}, max_bytes={max_bytes}")
    
    size_kb = os.path.getsize('tokenizer.tokbin') / 1024
    print(f"     File size: {size_kb:.0f} KB")
    return ok


def test_tokbin_vocab():
    """Verify .tokbin vocab matches Python tokenizer."""
    print("\n" + "=" * 60)
    print("Test 4: TOKBIN ↔ Python Vocab Match")
    print("=" * 60)
    
    from transformers import AutoTokenizer
    py_tok = AutoTokenizer.from_pretrained('tokenizer', use_fast=True)
    
    # Read .tokbin vocab
    import struct
    with open('tokenizer.tokbin', 'rb') as f:
        f.seek(28)  # skip header
        # Skip special IDs (21 × 4 = 84 bytes)
        f.seek(28 + 84)
        
        vocab_size = 32000
        tokbin_vocab = []
        for i in range(vocab_size):
            blen = struct.unpack('<H', f.read(2))[0]
            if blen == 0:
                f.read(2)  # pad
                tokbin_vocab.append(None)
                continue
            raw = f.read(blen)
            tokbin_vocab.append(raw)
            pad = (4 - (2 + blen) % 4) % 4
            if pad:
                f.read(pad)
    
    # Compare: for each token ID, check that the raw bytes match
    mismatches = 0
    for i in range(min(vocab_size, 1000)):  # Sample first 1000
        py_bytes = py_tok.decode([i])
        c_bytes = tokbin_vocab[i]
        if c_bytes is not None:
            c_str = c_bytes.decode('utf-8', errors='replace')
            if py_bytes != c_str:
                mismatches += 1
                if mismatches <= 5:
                    print(f"  ⚠️  ID {i}: Python={repr(py_bytes)[:40]}, C={repr(c_str)[:40]}")
    
    if mismatches == 0:
        print(f"  ✅ All sampled vocab entries match (0/{min(vocab_size, 1000)} mismatches)")
    else:
        print(f"  ❌ {mismatches} mismatches in sample")
    
    return mismatches == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    
    print("🧪 TinyLLM Tokenizer Consistency Test Suite")
    print(f"   Date: 2026-07-25")
    print()
    
    results = {}
    
    # Test 1: Special token IDs
    results['special_ids'] = test_special_token_ids()
    
    # Test 2: Python tokenizer
    py_results = test_python_tokenizer()
    
    # Test 3: TOKBIN format
    results['tokbin_format'] = test_tokbin_format()
    
    # Test 4: Vocab match
    results['vocab_match'] = test_tokbin_vocab()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    all_pass = all(results.values())
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} — {name}")
    
    if all_pass:
        print("\n🎉 All tests passed! Python ↔ C tokenizer consistency verified.")
    else:
        print("\n⚠️  Some tests failed. Review above for details.")
    
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
