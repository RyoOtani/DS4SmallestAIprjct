#!/usr/bin/env python3
"""
TinyLLM Tokenizer v7.1 — Real Data + Synthetic for 32K Vocab

Strategy:
  1. Download real code (CodeParrot) & wiki text via HuggingFace datasets
  2. Generate ~260K lines of varied synthetic code/text
  3. Use moderate multiplication ONLY if corpus < 10M chars (with warning)
  4. Fall back to aggressive synthetic if download fails

ByteLevel BPE needs ~50M+ chars for 32K vocab.
Target: ~15M chars of diverse corpus (×3 multiplier → 45M chars).
  
⚠️  QUALITY NOTE: Synthetic corpus with multiplier can cause vocabulary bias.
  For production use, provide real data via --corpus-dir or --corpus-file.
  Recommended: 50MB+ of real code + text for unbiased 32K vocab training.

Usage:
  python create_tokenizer.py                        # Default: synthetic
  python create_tokenizer.py --corpus-dir ./corpus  # Load .txt files
  python create_tokenizer.py --corpus-file data.txt # Single file
  python create_tokenizer.py --vocab 65536          # Larger vocab
"""

import json, os, sys, argparse, random, io, tempfile

random.seed(42)

# ═══════════════════════════════════════════════════════════════
# Data Collection
# ═══════════════════════════════════════════════════════════════

def download_real_data(max_mb=20):
    """Download real code + wiki text. Returns list of text lines."""
    lines = []
    
    # ── CodeParrot (GitHub code) ──
    try:
        from datasets import load_dataset
        print("  Downloading CodeParrot (streaming, ~10MB)...")
        ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True, trust_remote_code=True)
        count = 0
        for row in ds:
            code = row.get("content", "") or row.get("code", "") or ""
            if not code: continue
            for line in code.split("\n"):
                line = line.strip()
                if len(line) > 5 and len(line) < 500:
                    lines.append(line)
            count += 1
            if count % 100 == 0:
                mb = sum(len(l) for l in lines) / 1_000_000
                if mb > max_mb / 2: break
        print(f"    Got {len(lines):,} code lines (~{sum(len(l) for l in lines)/1_000_000:.1f}MB)")
    except Exception as e:
        print(f"    CodeParrot failed: {e}")

    # ── Wikipedia English ──
    try:
        from datasets import load_dataset
        print("  Downloading Wikipedia EN (streaming, ~5MB)...")
        ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True, trust_remote_code=True)
        count = 0
        for row in ds:
            text = row.get("text", "") or row.get("content", "") or ""
            if not text: continue
            for line in text.split("\n"):
                line = line.strip()
                if len(line) > 10 and len(line) < 1000:
                    lines.append(line)
            count += 1
            if count > 200: break
        print(f"    Total after wiki EN: {len(lines):,} lines")
    except Exception as e:
        print(f"    Wikipedia EN failed: {e}")

    # ── Wikipedia Japanese ──
    try:
        from datasets import load_dataset
        print("  Downloading Wikipedia JA (streaming, ~3MB)...")
        ds = load_dataset("wikipedia", "20220301.ja", split="train", streaming=True, trust_remote_code=True)
        count = 0
        for row in ds:
            text = row.get("text", "") or row.get("content", "") or ""
            if not text: continue
            for line in text.split("\n"):
                line = line.strip()
                if len(line) > 5 and len(line) < 1000:
                    lines.append(line)
            count += 1
            if count > 150: break
        print(f"    Total after wiki JA: {len(lines):,} lines")
    except Exception as e:
        print(f"    Wikipedia JA failed: {e}")

    return lines


def synthetic_corpus():
    """Heavy synthetic corpus — used as fallback or supplement."""
    random.seed(42)
    lines = []
    
    # ── Code: generate many unique identifiers and patterns ──
    py_kw = 'def class import from return yield async await if elif else for while break continue pass try except finally raise with as lambda True False None self super not and or in is'.split()
    py_types = 'int float str bytes bool list dict tuple set Optional Union Any Callable Iterable Sequence Mapping TypeVar Generator Iterator Coroutine'.split()
    py_libs = 'os sys json re math time datetime logging pathlib subprocess threading asyncio collections functools itertools dataclasses enum typing hashlib base64 uuid random string argparse numpy pandas torch flask fastapi pydantic sqlalchemy aiohttp httpx requests'.split()
    c_types = 'int char float double void long unsigned short size_t ssize_t uint8_t uint16_t uint32_t uint64_t int8_t int16_t int32_t int64_t bool uintptr_t ptrdiff_t'.split()
    rust_kw = 'fn struct enum impl trait pub use mod self mut ref let const static type dyn where async await match if else loop for while break continue return crate super'.split()
    go_kw = 'func type var const import package struct interface map chan select defer go range fallthrough break continue return'.split()
    js_kw = 'function const let var class extends import export default from async await return if else for while switch case break continue try catch throw new this super'.split()
    
    # Generate code lines: keyword + unique suffix
    for i, kw in enumerate(py_kw * 800):
        name = f"var_{i}_{random.randint(0,99999)}"
        if kw in ('def',):
            lines.append(f"{kw} {name}(x: {random.choice(py_types)}, y: {random.choice(py_types)} = None) -> {random.choice(py_types)}:")
            lines.append(f"    return {name}_impl(x, y)")
        elif kw in ('class',):
            lines.append(f"{kw} {name}({random.choice(py_types)}):")
            lines.append(f"    def __init__(self, value: {random.choice(py_types)}): self.value = value")
        elif kw in ('import',):
            lib = random.choice(py_libs)
            lines.append(f"{kw} {lib}")
            lines.append(f"from {lib} import {random.choice(py_kw[:20])}")
        elif kw in ('for',):
            lines.append(f"{kw} item in range({random.randint(0,1000)}): process(item)")
        elif kw in ('if',):
            lines.append(f"{kw} value > {random.randint(0,100)}: return {random.choice(['True','False']) if random.random()<0.5 else random.randint(0,999)}")
        elif kw in ('try',):
            lines.append(f"{kw}: result = await operation_{i}()")
            lines.append(f"except Exception as e: logger.error(f'failed: {{e}}')")
        elif kw in ('return',):
            lines.append(f"{kw} {name} if {name} is not {random.choice(['None','False','0','[]'])} else default_value")
        elif kw in ('with',):
            lines.append(f"{kw} open(f'data_{i}.json') as f: payload = json.load(f)")
        elif kw in ('async',):
            lines.append(f"{kw} def handler_{i}(request: Request) -> Response:")
        elif kw in ('yield',):
            lines.append(f"{kw} from gen_{i}(data_{i})")
        elif kw in ('assert',):
            lines.append(f"{kw} isinstance(value_{i}, {random.choice(py_types)}), f'Expected {random.choice(py_types)}'")
        elif kw in ('raise',):
            lines.append(f"{kw} {random.choice(['ValueError','TypeError','RuntimeError','NotImplementedError'])}('{name} failed at step {i}')")
        elif kw in ('lambda',):
            lines.append(f"fn = {kw} x: x.{random.choice(['upper','lower','strip','split','replace','encode','decode'])}()")
        else:
            lines.append(f"    {kw} {name} = compute_{i}(input_{i})")
    
    # C lines
    for i, t in enumerate(c_types * 200):
        lines.append(f"{t} compute_{i}_{t}({t} x, const {t}* buf, size_t n) {{")
        lines.append(f"    {t} acc = 0;")
        lines.append(f"    for (size_t j = 0; j < n; j++) acc += buf[j];")
        lines.append(f"    return acc;")
        lines.append(f"}}")
    
    # Rust lines
    for i, kw in enumerate(rust_kw * 400):
        lines.append(f"{kw} {kw}_{i}_{random.randint(0,999)};")
    
    # Go lines
    for i, kw in enumerate(go_kw * 500):
        lines.append(f"{kw} {kw}_{i}_{random.randint(0,999)};")
    
    # JS lines
    for i, kw in enumerate(js_kw * 400):
        lines.append(f"{kw} {kw}_{i}_{random.randint(0,999)};")
    
    # ── English: varied templates with unique parameters ──
    en_words = 'algorithm architecture system component function module library framework pipeline service deployment configuration authentication authorization encryption optimization performance scalability reliability availability maintainability latency throughput bandwidth overhead benchmark buffer cache compiler container database debug endpoint interface iteration kernel load middleware namespace parameter protocol query recursion repository schema semaphore serialization session specification synchronization tokenizer transaction validation variable virtualization vulnerability heuristic abstraction encapsulation inheritance polymorphism dependency concurrency parallelism asynchrony idempotency determinism'.split()
    en_verbs = 'improved optimized refactored enhanced simplified redesigned overhauled streamlined restructured reorganized modernized upgraded extended integrated consolidated standardized'.split()
    en_adjs = 'essential critical crucial vital fundamental important necessary significant valuable beneficial advantageous powerful robust flexible scalable efficient effective reliable secure maintainable extensible portable interoperable'.split()
    
    for i in range(50000):
        w1, w2, w3 = random.sample(en_words, 3)
        verb = random.choice(en_verbs)
        adj = random.choice(en_adjs)
        lines.append(f"The {w1} {w2} has been {verb} with {adj} {w3} in version {random.randint(1,20)}.{random.randint(0,99)}.")
        lines.append(f"Proper {w1} implementation is {adj} for building {adj} {w2} systems.")
        if i % 3 == 0:
            lines.append(f"We need to {verb} the {w1} module to handle {w2} more {random.choice(['efficiently','effectively','securely','reliably'])}.")
        if i % 5 == 0:
            lines.append(f"Have you considered using {w1} instead of {w2} for the {w3} component?")
        if i % 7 == 0:
            lines.append(f"The integration of {w1} and {w2} requires careful {w3} management.")
        if i % 11 == 0:
            lines.append(f"Please review the {w1} documentation before proceeding with the {w2} deployment at {random.randint(1,12)}:{random.choice(['00','15','30','45'])} {random.choice(['AM','PM'])}.")
    
    # ── Japanese: varied templates ──
    jp_nouns = 'アルゴリズム アーキテクチャ システム コンポーネント 機能 モジュール ライブラリ フレームワーク パイプライン サービス デプロイ 設定 認証 認可 暗号化 最適化 パフォーマンス スケーラビリティ 信頼性 可用性 保守性 レイテンシ スループット 帯域幅 オーバーヘッド ベンチマーク バッファ キャッシュ コンパイラ コンテナ データベース デバッグ エンドポイント インターフェース イテレーション カーネル ロード ミドルウェア 名前空間 パラメータ プロトコル クエリ 再帰 リポジトリ スキーマ セマフォ シリアライゼーション セッション 仕様 同期 トークナイザー トランザクション バリデーション 変数 仮想化 脆弱性 継承 カプセル化 ポリモーフィズム 依存性 並行性 並列性'.split()
    jp_verbs = '改善 最適化 強化 簡素化 再設計 刷新 合理化 再構築 再編成 近代化 アップグレード 拡張 統合 統合化 標準化'.split()
    
    for i in range(30000):
        w1, w2, w3 = random.sample(jp_nouns, 3)
        verb = random.choice(jp_verbs)
        lines.append(f"バージョン{random.randint(1,20)}.{random.randint(0,99)}において{w1}の{w2}が{verb}されました。")
        lines.append(f"適切な{w1}の実装は{w2}システムの構築に不可欠です。")
        if i % 4 == 0:
            lines.append(f"先日の{w1}に関する会議で議論された{w2}について追加の確認をお願いします。")
        if i % 6 == 0:
            lines.append(f"{w1}と{w2}の統合には慎重な{w3}管理が求められます。")
        if i % 9 == 0:
            lines.append(f"新しい{w1}のリリースノートを確認しました。{w2}の改善が素晴らしいですね。")
    
    # ── Misc: paths, configs, commands ──
    for i in range(10000):
        lines.append(f"src/module_{i}/file_{i%100}.{random.choice(['py','js','ts','rs','go','c','h','json','yaml','md'])}")
        lines.append(f"https://api.service_{i%50}.com/v{random.randint(1,3)}/endpoint_{i}")
    
    for i in range(5000):
        lines.append(f"export VAR_{i}=value_{i}_{random.randint(1000,9999)}")
        lines.append(f"--{random.choice(['config','output','input','format','debug','verbose','quiet','force','yes','no'])} value_{i}")
    
    return list(dict.fromkeys(lines))


# ═══════════════════════════════════════════════════════════════
# Tokenizer
# ═══════════════════════════════════════════════════════════════

def create_tokenizer(output_dir="tokenizer", vocab_size=32000, corpus_dir=None, corpus_file=None):
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
        from tokenizers.normalizers import NFKC
    except ImportError:
        print("pip install tokenizers")
        return None

    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print(f"TinyLLM Tokenizer v7 — Target vocab={vocab_size:,}")
    print("=" * 60)

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.normalizer = NFKC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[("<s>", 0), ("</s>", 1)],
    )

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

    # Build corpus
    print("\nBuilding corpus...")
    real = download_real_data()
    synthetic = synthetic_corpus()
    
    # Load external corpus files if provided
    external = []
    if corpus_dir and os.path.isdir(corpus_dir):
        for fname in sorted(os.listdir(corpus_dir)):
            fpath = os.path.join(corpus_dir, fname)
            if fname.endswith(('.txt', '.py', '.js', '.ts', '.rs', '.go', '.c', '.h', '.md')):
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if 5 < len(line) < 1000:
                                external.append(line)
                except: pass
        print(f"  Corpus dir:  {len(external):,} lines from {corpus_dir}")
    if corpus_file and os.path.isfile(corpus_file):
        try:
            with open(corpus_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if 5 < len(line) < 1000:
                        external.append(line)
            print(f"  Corpus file: {len(external):,} lines from {corpus_file}")
        except: pass
    
    corpus = real + synthetic + external
    random.shuffle(corpus)
    
    total_chars = sum(len(l) for l in corpus)
    print(f"  Real data:  {len(real):,} lines")
    print(f"  Synthetic:  {len(synthetic):,} lines")
    print(f"  External:   {len(external):,} lines")
    print(f"  Total:      {len(corpus):,} lines (~{total_chars/1_000_000:.1f}M chars)")

    # ByteLevel BPE requires sufficient corpus to discover merge rules.
    # If corpus is too small, moderate repetition is acceptable —
    # but excessive repetition causes vocabulary bias toward repeated patterns.
    # Best practice: provide 50MB+ of real text via --corpus-dir.
    MIN_CHARS = 30_000_000  # 30M chars minimum for 32K vocab
    if total_chars < MIN_CHARS:
        multiplier = min(int(MIN_CHARS / max(total_chars, 1)), 15)
        if multiplier > 1:
            print(f"  ⚠️  Corpus too small ({total_chars/1e6:.1f}M chars < {MIN_CHARS/1e6:.0f}M)")
            print(f"     Using x{multiplier} repetition (may cause slight vocabulary bias)")
            print(f"     For production: add real data with --corpus-dir ./my_texts/")
            corpus = corpus * multiplier
            print(f"     → {len(corpus):,} lines (~{total_chars*multiplier/1_000_000:.1f}M chars)")
    else:
        print(f"  ✅ Corpus sufficient ({total_chars/1e6:.1f}M chars), no repetition needed")

    # Train
    print(f"\nTraining BPE (vocab={vocab_size:,}, min_freq=1)...")
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=1,
    )
    tok.train_from_iterator(corpus, trainer)
    tok.save(f"{output_dir}/tokenizer.json")
    actual = tok.get_vocab_size()

    # HF config
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
    print(f"\n✅ Done! {kb:.0f} KB, vocab={actual:,} / {vocab_size:,}")
    if actual < vocab_size * 0.8:
        print(f"⚠ Only reached {actual/vocab_size:.0%} of target. Need more data variety.")
    return tok


def test_tokenizer(output_dir="tokenizer"):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("transformers not installed. Run: pip install transformers")
        return
    tok = AutoTokenizer.from_pretrained(output_dir, use_fast=True)
    tests = [
        ("Python-algo", "def binary_search(arr: list[int], target: int) -> int:\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: lo = mid + 1\n        else: hi = mid - 1\n    return -1"),
        ("C-func", "#include <stdio.h>\nint factorial(int n) { return n <= 1 ? 1 : n * factorial(n - 1); }\nint main(void) { printf(\"%d\\n\", factorial(5)); return 0; }"),
        ("JS-React", "import { useState, useEffect } from 'react';\nconst Counter: FC = () => {\n  const [count, setCount] = useState(0);\n  useEffect(() => { document.title = `Count: ${count}`; }, [count]);\n  return <button onClick={() => setCount(c => c + 1)}>+1</button>;\n};"),
        ("Rust-srv", "use tokio::net::TcpListener;\n#[tokio::main]\nasync fn main() -> Result<()> {\n    let listener = TcpListener::bind(\"0.0.0.0:8080\").await?;\n    loop {\n        let (stream, addr) = listener.accept().await?;\n        tokio::spawn(async move { handle(stream).await });\n    }\n}"),
        ("Go-srv", "func (s *Server) handleUsers(w http.ResponseWriter, r *http.Request) {\n    w.Header().Set(\"Content-Type\", \"application/json\")\n    if r.Method != http.MethodGet {\n        http.Error(w, \"not allowed\", 405)\n        return\n    }\n}"),
        ("SQL", "SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id WHERE u.active = true GROUP BY u.id HAVING COUNT(o.id) > 5 ORDER BY COUNT(o.id) DESC LIMIT 10;"),
        ("EN-tech", "The transformer architecture employs multi-head self-attention with scaled dot-product similarity to process sequences in parallel."),
        ("EN-chat", "Hey, I'm debugging an issue with the authentication middleware — could you take a quick look when you have a moment?"),
        ("JP-tech", "トランスフォーマーモデルは自己注意機構を用いて文脈を理解し、高精度なテキスト生成を実現します。"),
        ("JP-chat", "お疲れ様です。先日のバグ修正について確認したいことがあるのですが、今お時間よろしいでしょうか。"),
        ("Config", "DATABASE_URL=postgresql://user:password@localhost:5432/mydb\nREDIS_URL=redis://localhost:6379\nJWT_SECRET=super-secret-value-change-me-in-production"),
        ("Shell", "docker build -t myapp:latest . && docker push registry.example.com/myapp:latest && kubectl apply -f k8s/deployment.yaml"),
        ("FIM", "<fim_prefix>def quick_sort(arr):<fim_suffix>    return arr<fim_middle>    if len(arr) <= 1: return arr\n    pivot = arr[0]\n    left = [x for x in arr[1:] if x <= pivot]\n    right = [x for x in arr[1:] if x > pivot]\n    return quick_sort(left) + [pivot] + quick_sort(right)"),
        ("Unicode", "こんにちは世界！🚀 Hello 世界 🌍 Привет мир 日本語 한국어 中文 ✨ αβγ ∑∫ √∞"),
    ]
    print("=" * 80)
    print(f"{'Test':12s} {'Chars':>5} {'Tokens':>6} {'Ratio':>6}  Sample")
    print("=" * 80)
    for label, text in tests:
        ids = tok.encode(text)
        ratio = len(ids) / max(len(text), 1)
        pieces = [tok.decode([i]) for i in ids[:4]]
        print(f"  {label:10s}  {len(text):5d}  {len(ids):6d}  {ratio:5.2f}x  {' | '.join(pieces)}")
    print("=" * 80)
    print(f"  Vocab: {tok.vocab_size:,}  |  ByteLevel BPE  |  FIM  |  Code×EN×JP")
    print("=" * 80)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='TinyLLM Tokenizer v7.1')
    p.add_argument('--test', action='store_true', help='Test after building')
    p.add_argument('--output', default='tokenizer', help='Output directory')
    p.add_argument('--vocab', type=int, default=32000, help='Target vocab size')
    p.add_argument('--corpus-dir', default=None, help='Directory with .txt/.py/.js etc files')
    p.add_argument('--corpus-file', default=None, help='Single text file for corpus')
    args = p.parse_args()
    create_tokenizer(args.output, args.vocab, args.corpus_dir, args.corpus_file)
    if args.test:
        test_tokenizer(args.output)
