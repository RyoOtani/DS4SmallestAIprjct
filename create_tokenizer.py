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
    """Generate training data for the BPE tokenizer."""
    languages = ['python', 'c', 'javascript', 'rust', 'go', 'typescript']

    samples = [
        # Python
        "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr)//2]",
        "class Node:\n    def __init__(self, val):\n        self.val = val\n        self.left = None",
        "import asyncio\nasync def fetch(url):\n    async with aiohttp.ClientSession() as s:",
        "@dataclass\nclass Config:\n    learning_rate: float = 1e-4\n    batch_size: int = 32",
        "try:\n    result = await process(data)\nexcept ValueError as e:\n    log.error(e)",

        # C/C++
        "#include <stdio.h>\nint main() {\n    printf(\"Hello, World!\\n\");\n    return 0;\n}",
        "struct Node { int data; struct Node* next; };\nvoid insert(struct Node** head, int val) {",
        "#include <vector>\nstd::vector<int> v = {1, 2, 3, 4, 5};\nstd::sort(v.begin(), v.end());",
        "void* malloc(size_t size);\nvoid free(void* ptr);\n#define MAX(a,b) ((a)>(b)?(a):(b))",

        # JavaScript/TypeScript
        "function fibonacci(n) { if (n <= 1) return n; return fibonacci(n-1) + fibonacci(n-2); }",
        "const result = [1, 2, 3, 4, 5].filter(x => x % 2 === 0).map(x => x * x);",
        "class EventEmitter { constructor() { this.events = {}; } on(event, fn) {",
        "interface User { id: number; name: string; email: string; }",
        "async function fetchData<T>(url: string): Promise<T> { const res = await fetch(url);",

        # Rust
        "fn main() { let v = vec![1, 2, 3]; for x in &v { println!(\"{}\", x); } }",
        "impl<T: Clone + Ord> BinaryHeap<T> { fn new() -> Self { Self { data: Vec::new() } } }",
        "#[derive(Debug, Clone, Serialize, Deserialize)]\npub struct Config { pub port: u16 }",

        # Go
        "package main\nimport \"fmt\"\nfunc main() { fmt.Println(\"Hello\"); }",
        "type User struct { Name string `json:\"name\"`; Age int `json:\"age\"` }",
        "func (s *Server) HandleRequest(w http.ResponseWriter, r *http.Request) {",

        # General text
        "The transformer architecture uses self-attention to process sequences.",
        "Gradient descent is an optimization algorithm that minimizes the loss function.",
        "A binary search tree maintains the invariant that left < root < right.",
        "HTTP/1.1 200 OK\nContent-Type: application/json\n\n{\"status\": \"ok\"}",
        "git commit -m \"Add feature\" && git push origin main",
        "SELECT * FROM users WHERE email LIKE '%@example.com' ORDER BY created_at DESC",
    ]

    # Duplicate and shuffle to create enough data for BPE training
    import itertools
    for i, sample in enumerate(itertools.cycle(samples)):
        if i >= 5000:
            break
        yield sample


if __name__ == '__main__':
    create_tokenizer()
