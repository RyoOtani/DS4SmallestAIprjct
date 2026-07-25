#!/usr/bin/env python3
"""
export_tokenizer.py — Convert tokenizer.json → .tokbin for C runtime

Uses the `tokenizers` library to decode ByteLevel BPE tokens to raw bytes,
producing a simple binary format (.tokbin) that C can load with zero deps.

Binary format (.tokbin) — all little-endian:
  Header (28B): magic[4] version[4] vocab_size[4] n_merges[4]
                n_specials[4] max_token_bytes[4] flags[4]
  Special IDs:  special_ids[n_specials × u32]
  Vocab:        for each ID 0..vocab_size-1:
                  byte_len[u16] raw_bytes[byte_len] pad_to_4
                (empty entry = byte_len=0, pad=2B)
  Merges:       for each merge: a[u32] b[u32] rank[u32]

Usage: python export_tokenizer.py [--input tokenizer/tokenizer.json]
"""

import json, struct, os, sys

TOKBIN_MAGIC = 0x424B4F54  # "TOKB"
TOKBIN_VERSION = 2

SPECIAL_NAMES = [
    "<s>", "</s>", "<pad>", "<unk>",
    "<fim_prefix>", "<fim_suffix>", "<fim_middle>",
    "<fim_hole>", "<fim_pad>",
    "<repo_name>", "<file_sep>", "<file_path>",
    "<tool_call>", "</tool_call>",
    "<tool_response>", "</tool_response>",
    "<scratchpad>", "</scratchpad>",
    "<|system|>", "<|user|>", "<|assistant|>",
]
NUM_SPECIALS = len(SPECIAL_NAMES)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='tokenizer/tokenizer.json')
    p.add_argument('--output', default='tokenizer.tokbin')
    args = p.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ {args.input} not found"); sys.exit(1)

    print(f"📥 Loading: {args.input}")
    with open(args.input, 'rb') as f:
        data = json.load(f)

    model = data.get('model', {})
    raw_vocab = model.get('vocab', {})        # {str: int}
    raw_merges = model.get('merges', [])       # [["a","b"], ...]

    # Get special token IDs from tokenizer_config.json
    cfg_path = os.path.join(os.path.dirname(args.input), 'tokenizer_config.json')
    special_ids = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        for id_str, info in cfg.get('added_tokens_decoder', {}).items():
            if info.get('special', False):
                special_ids[info['content']] = int(id_str)

    # Use tokenizers library for ByteLevel decoding
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    bld = ByteLevelDecoder()

    # Build vocab: token_id → raw_bytes
    max_id = max(raw_vocab.values()) + 1
    vocab_bytes = [None] * max_id
    for display_str, tok_id in raw_vocab.items():
        decoded = bld.decode([display_str])
        vocab_bytes[tok_id] = decoded.encode('utf-8')

    # Build merge table from raw_merges (list of [str, str] pairs)
    merge_table = []
    for rank, pair in enumerate(raw_merges):
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        a_id = raw_vocab.get(pair[0])
        b_id = raw_vocab.get(pair[1])
        if a_id is None: a_id = special_ids.get(pair[0])
        if b_id is None: b_id = special_ids.get(pair[1])
        if a_id is not None and b_id is not None:
            merge_table.append((a_id, b_id, rank))

    # Print special token ID verification
    print(f"\n📋 Special token IDs:")
    for i, name in enumerate(SPECIAL_NAMES):
        tid = special_ids.get(name, -1)
        mark = "✅" if tid == i else "❌ MISMATCH"
        print(f"   {mark}  ID {tid:>5d} = {name}")

    max_token_bytes = max((len(b) for b in vocab_bytes if b), default=4)

    # Write binary
    with open(args.output, 'wb') as f:
        f.write(struct.pack('<I', TOKBIN_MAGIC))
        f.write(struct.pack('<I', TOKBIN_VERSION))
        f.write(struct.pack('<I', len(vocab_bytes)))
        f.write(struct.pack('<I', len(merge_table)))
        f.write(struct.pack('<I', NUM_SPECIALS))
        f.write(struct.pack('<I', max_token_bytes))
        f.write(struct.pack('<I', 0))

        for name in SPECIAL_NAMES:
            f.write(struct.pack('<I', special_ids.get(name, 0xFFFFFFFF)))

        for raw in vocab_bytes:
            if raw is None:
                f.write(struct.pack('<H', 0))
                f.write(struct.pack('<H', 0))
            else:
                blen = len(raw)
                f.write(struct.pack('<H', blen))
                f.write(raw)
                pad = (4 - (2 + blen) % 4) % 4
                if pad:
                    f.write(b'\x00' * pad)

        for a_id, b_id, rank in merge_table:
            f.write(struct.pack('<III', a_id, b_id, rank))

    size_kb = os.path.getsize(args.output) / 1024
    eff = sum(1 for b in vocab_bytes if b)
    print(f"\n✅ {args.output} ({size_kb:.0f} KB)")
    print(f"   Vocab: {eff:,}/{len(vocab_bytes):,} | Merges: {len(merge_table):,} | Max bytes: {max_token_bytes}")


if __name__ == '__main__':
    main()
