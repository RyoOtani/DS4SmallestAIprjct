#!/usr/bin/env python3
"""
create_tokenizer_small.py — English + Japanese + Code specialized tokenizer.

Design:
  - BPE (Byte-Pair Encoding) tokenizer
  - Vocab size: 72,000 (configurable 65K-80K)
  - Specialized corpus: English (docs, forums), Japanese (wiki, tech),
    Code (Python, JS, Rust, C, Shell)
  - References nano model's 32K tokenizer structure
  - Compatible with HuggingFace tokenizers library

Usage:
  python create_tokenizer_small.py                    # Create tokenizer with default settings
  python create_tokenizer_small.py --vocab 80000      # 80K vocab
  python create_tokenizer_small.py --output my_tok    # Custom output dir
"""
import os, sys, json, argparse, random
from pathlib import Path
from typing import List

# Add repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ═══════════════════════════════════════════════════════
# Synthetic Multilingual + Code Corpus Generator
# ═══════════════════════════════════════════════════════

def generate_diverse_corpus(num_lines: int = 300000) -> List[str]:
    """
    Generate a highly diverse corpus covering many byte patterns.
    Mix: real-looking code, varied natural language, random strings, Unicode.
    """
    import itertools, string as str_mod

    lines = []

    # ── 1. Real-looking code in many languages ──
    python_keywords = ["def", "class", "import", "from", "async", "await", "yield",
                       "lambda", "with", "try", "except", "raise", "return", "if",
                       "elif", "else", "for", "while", "break", "continue", "pass",
                       "assert", "global", "nonlocal", "del", "in", "is", "not", "or", "and"]
    python_types = ["int", "str", "float", "bool", "list", "dict", "tuple", "set",
                    "Optional", "Union", "Any", "Callable", "TypeVar", "Generic",
                    "Iterable", "Sequence", "Mapping", "Coroutine", "Awaitable"]

    rust_keywords = ["fn", "let", "mut", "pub", "impl", "struct", "enum", "trait",
                     "match", "loop", "where", "async", "await", "move", "ref",
                     "unsafe", "extern", "crate", "mod", "use", "self", "super",
                     "dyn", "const", "static", "type", "Box", "Vec", "Option", "Result"]

    # Generate diverse code snippets
    for _ in range(num_lines // 4):
        # Python code
        indent = "    "
        func_name = ''.join(random.choice(str_mod.ascii_lowercase) for _ in range(random.randint(4, 12)))
        arg_names = [''.join(random.choice(str_mod.ascii_lowercase) for _ in range(random.randint(1, 6)))
                     for _ in range(random.randint(0, 3))]
        ret_type = random.choice(python_types)
        body_lines = [
            f"{indent}{random.choice(python_keywords)} {random.choice(arg_names) if arg_names else 'x'}",
            f"{indent}result = {random.choice(arg_names) if arg_names else 'data'}.{random.choice(['get','process','compute','transform','validate','parse','encode','decode','format'])}({', '.join(random.choices(arg_names, k=min(2,len(arg_names)))) if arg_names else ''})",
            f"{indent}logger.{random.choice(['info','debug','warning','error'])}(f\"{{result}}\")",
        ]
        code = f"def {func_name}({', '.join(f'{a}: {random.choice(python_types)}' for a in arg_names)}) -> {ret_type}:\n"
        code += f"{indent}\"\"\"{random.choice(['Process data','Handle request','Compute result','Validate input'])}.\"\"\"\n"
        code += '\n'.join(random.sample(body_lines, min(3, len(body_lines))))
        code += f"\n{indent}return {random.choice(['result','True','None','data'])}"
        lines.append(code)

        # Rust code
        lines.append(f"fn {func_name}({', '.join(f'{a}: {random.choice(python_types)}' for a in arg_names)}) -> Result<{ret_type}> {{\n    let {random.choice(arg_names) if arg_names else 'val'} = {random.choice(rust_keywords)};\n    Ok({random.choice(arg_names) if arg_names else 'val'})\n}}")

        # TypeScript
        lines.append(f"const {func_name} = ({', '.join(f'{a}: {t}' for a,t in zip(arg_names, random.choices(python_types, k=len(arg_names))))}): {ret_type} => {{\n  const result = {random.choice(arg_names) if arg_names else 'input'}.{random.choice(['map','filter','reduce','find','some','every','flatMap','slice','concat'])}(item => item.id);\n  return result;\n}};")

        # SQL
        cols = random.sample(["id", "name", "email", "created_at", "status", "score", "metadata", "token_count", "priority", "category"], random.randint(2, 6))
        lines.append(f"SELECT {', '.join(cols)} FROM {random.choice(['users','orders','products','sessions','events','logs','items','tasks'])} WHERE {random.choice(cols)} > ${random.randint(1,100)} AND status = '{random.choice(['active','pending','completed','failed','archived'])}' ORDER BY {random.choice(cols)} DESC LIMIT {random.randint(10,100)};")

        # Shell
        lines.append(f"find {random.choice(['/src','/data','/logs','/config','/lib'])} -name \"*.{random.choice(['py','ts','rs','js','go','java','rb'])}\" -exec grep -l \"{random.choice(python_keywords)}\" {{}} \\; | head -n {random.randint(5,50)} | while read f; do wc -l \"$f\"; done | sort -rn")

    # ── 2. Diverse natural language ──
    nouns = ["algorithm", "architecture", "component", "database", "endpoint",
             "framework", "generator", "handler", "interface", "journal",
             "kernel", "library", "module", "network", "operator",
             "protocol", "query", "router", "scheduler", "template",
             "utility", "validator", "worker", "parser", "encoder",
             "decoder", "serializer", "cache", "queue", "stream",
             "token", "session", "payload", "header", "response",
             "request", "middleware", "controller", "model", "view",
             "service", "repository", "factory", "builder", "proxy",
             "adapter", "bridge", "facade", "observer", "strategy",
             "command", "iterator", "visitor", "decorator", "singleton"]
    verbs = ["process", "handle", "compute", "generate", "validate",
             "transform", "optimize", "configure", "initialize", "execute",
             "deploy", "monitor", "analyze", "aggregate", "serialize",
             "deserialize", "encrypt", "decrypt", "authenticate", "authorize",
             "provision", "orchestrate", "schedule", "dispatch", "route"]

    for _ in range(num_lines // 4):
        # English
        lines.append(f"The {random.choice(nouns)} {random.choice(verbs)}s the {random.choice(nouns)} using a {random.choice(nouns)}-based {random.choice(nouns)} with {random.randint(2,16)} {random.choice(nouns)}s.")
        lines.append(f"To {random.choice(verbs)} the {random.choice(nouns)}, configure the {random.choice(nouns)}.{random.choice(nouns)} parameter to {random.choice(['true','false','auto','manual','dynamic','static','lazy','eager','async','sync'])}.")
        lines.append(f"Benchmark results: {random.choice(nouns)} achieved {random.randint(100,99999)} {random.choice(['ops/sec','ms latency','MB/s','req/s','tokens/s'])} with {random.randint(1,99)}% {random.choice(['improvement','regression','overhead','efficiency'])}.")

        # Japanese
        ja_nouns = ["システム", "データ", "コード", "エラー", "パフォーマンス",
                     "アルゴリズム", "データベース", "API", "フレームワーク", "テスト",
                     "デプロイ", "最適化", "セキュリティ", "認証", "キャッシュ",
                     "メモリ", "ネットワーク", "プロトコル", "コンテナ", "サーバー"]
        ja_verbs = ["処理する", "実装する", "設計する", "最適化する", "検証する",
                     "監視する", "分析する", "構成する", "実行する", "変換する"]
        ja_adj = ["高速な", "効率的な", "安全な", "安定した", "柔軟な",
                   "堅牢な", "軽量な", "拡張性のある", "信頼性の高い", "再利用可能な"]
        lines.append(f"{random.choice(ja_adj)}{random.choice(ja_nouns)}を{random.choice(ja_verbs)}方法について解説します。")
        lines.append(f"この{random.choice(ja_nouns)}は{random.randint(10,999)}個の{random.choice(ja_nouns)}を{random.choice(ja_verbs)}ことができます。")
        lines.append(f"{random.choice(ja_nouns)}の{random.choice(ja_verbs)}に関する{random.randint(1,20)}のベストプラクティス。")

    # ── 3. Random diverse strings (covers rare byte pairs) ──
    for _ in range(num_lines // 4):
        # Random hex strings
        lines.append(''.join(random.choice('0123456789abcdef') for _ in range(random.randint(8, 64))))
        # Random base64-like
        lines.append(''.join(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=') for _ in range(random.randint(8, 128))))
        # JSON-like fragments
        lines.append('{' + ', '.join(f'"{random.choice(nouns)}": {random.choice(["true","false","null",str(random.randint(0,9999)),f'"{random.choice(nouns)}"'])}' for _ in range(random.randint(2, 8))) + '}')
        # URL paths
        paths = ['/'.join(random.sample(nouns, random.randint(2,5))) for _ in range(1)]
        lines.append(f"https://{random.choice(nouns)}.example.com/{paths[0]}?{random.choice(nouns)}={random.randint(1,999)}&{random.choice(nouns)}={random.choice(['true','false','auto'])}")
        # Log lines
        lines.append(f"[{random.randint(2000,2026)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}T{random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}.{random.randint(0,999):03d}Z] {random.choice(['INFO','DEBUG','WARN','ERROR','TRACE'])} {random.choice(nouns)}.{random.choice(verbs)} - {random.choice(nouns)}={random.choice(['ok','fail','timeout','pending'])} duration={random.randint(1,9999)}ms")

    # ── 4. Unicode and special characters ──
    unicode_samples = [
        "αβγδεζηθικλμνξπρστυφχψω",  # Greek
        "∀∃∄∅∆∇∈∉∊∋∌∍∎∏∐∑−∓∔∕∖∗∘∙√∛∜∝∞∟",  # Math symbols
        "←↑→↓↔↕↖↗↘↙↚↛↜↝↞↟",  # Arrows
        "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳",  # Circled numbers
        "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん",
        "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン",
        "가각간갇갈감갑값갓갔강갖갗같갚갛개객갠갤갬갭갯갰갱갸갹갼걀걁걂걃걄걅걆걇걈걉걊걋걌걍걎걏",
    ]
    for sample in unicode_samples:
        for _ in range(num_lines // 100):
            lines.append(sample[:random.randint(5, len(sample))])

    random.shuffle(lines)
    return lines[:num_lines]
    """English technical + general text."""
    templates = [
        # Technical documentation
        "The {component} uses a {algorithm} to process {data} efficiently.",
        "To implement {feature}, you need to configure the {setting} parameter.",
        "The {function} returns a {type} that represents the {concept}.",
        "When {condition} occurs, the system triggers a {response} mechanism.",
        "Optimization of {process} reduced latency by {percent} percent.",
        "The API endpoint {endpoint} accepts {method} requests with {content_type} payload.",
        "For large-scale {workload}, consider using distributed {architecture}.",
        "The {framework} provides built-in support for {protocol} communication.",
        "Error handling in {language} follows the {pattern} pattern for robustness.",
        "Database {operation} performance depends on {index_type} indexing strategy.",

        # General conversation
        "I think the best approach would be to {action} before {action2}.",
        "Could you explain how {concept} relates to {concept2}?",
        "The main difference between {thing1} and {thing2} is the {aspect}.",
        "Let me show you an example of how to use {tool} for {task}.",
        "According to the documentation, {feature} was introduced in version {version}.",
        "The community has developed several {alternatives} to address this issue.",
        "Performance benchmarks show that {method1} outperforms {method2} by {percent}%.",
        "Security considerations for {system} include {vulnerability} prevention.",
        "The {design} pattern is commonly used in {domain} applications.",
        "Testing {component} requires mocking the {dependency} interface.",
    ]

    words = {
        "component": ["cache", "router", "parser", "encoder", "scheduler", "dispatcher", "validator"],
        "algorithm": ["hashing", "sorting", "graph traversal", "dynamic programming", "greedy"],
        "data": ["streaming data", "batch inputs", "user requests", "log entries", "sensor readings"],
        "feature": ["authentication", "rate limiting", "caching", "logging", "compression"],
        "function": ["process_request", "validate_input", "compute_hash", "serialize_object"],
        "type": ["dictionary", "list", "optional string", "integer array", "generic type"],
        "concept": ["dependency injection", "lazy evaluation", "memoization", "polymorphism"],
        "language": ["Python", "Rust", "TypeScript", "Go", "C++", "Java", "Kotlin"],
        "framework": ["React", "Django", "FastAPI", "Axum", "Next.js", "Spring Boot"],
        "protocol": ["HTTP/2", "WebSocket", "gRPC", "MQTT", "AMQP", "GraphQL"],
        "architecture": ["microservices", "event-driven", "CQRS", "hexagonal", "layered"],
        "pattern": ["Result type", "Circuit Breaker", "Retry with backoff", "Decorator"],
        "method": ["gradient descent", "random forest", "convolution", "attention mechanism"],
        "tool": ["Docker", "Kubernetes", "Terraform", "GitHub Actions", "Prometheus"],
    }

    lines = []
    for _ in range(num_lines):
        tmpl = random.choice(templates)
        for key, values in words.items():
            if "{" + key + "}" in tmpl:
                tmpl = tmpl.replace("{" + key + "}", random.choice(values), 1)
        # Fill remaining placeholders
        for key in ["setting", "condition", "response", "process", "percent",
                     "endpoint", "content_type", "workload", "operation", "index_type",
                     "action", "action2", "concept2", "thing1", "thing2", "aspect",
                     "task", "version", "alternatives", "method1", "method2",
                     "system", "vulnerability", "design", "domain", "dependency"]:
            tmpl = tmpl.replace("{" + key + "}", random.choice(["default", "primary", "secondary", "main", "core", "auxiliary"]), 1)
        lines.append(tmpl)

    return lines


def generate_japanese_corpus(num_lines: int = 60000) -> List[str]:
    """Japanese technical + general text."""
    templates = [
        # Technical
        "{technology}の{aspect}について詳しく説明してください。",
        "この{component}は{feature}を提供するために{method}を使用しています。",
        "{language}での{task}の実装方法を教えてください。",
        "{framework}を使った{application}の開発手順を解説します。",
        "最新の{field}研究では{approach}が注目されています。",
        "{system}のパフォーマンスを{percent}%向上させる方法。",
        "エラー「{error_message}」の解決方法を教えてください。",
        "{database}で{query}を最適化するテクニック。",
        "{algorithm}の時間計算量はO({complexity})です。",
        "{tool}を使用して{process}を自動化する手順。",

        # General
        "{topic}についての意見を聞かせてください。",
        "最近{trend}が{industry}業界で話題になっています。",
        "{location}での{event}に参加してきました。",
        "{book}を読んで{concept}の理解が深まりました。",
        "{person}の{work}は本当に素晴らしいと思います。",
        "{year}年の{category}トレンドをまとめました。",
        "この{problem}に対する{strategy}を検討しています。",
        "{skill}を習得するためのおすすめの学習リソース。",
        "{company}の{product}が{market}でシェアを拡大中です。",
    ]

    words = {
        "technology": ["機械学習", "ブロックチェーン", "量子コンピューティング", "エッジAI"],
        "aspect": ["アーキテクチャ", "設計思想", "実装方法", "パフォーマンス特性", "セキュリティ面"],
        "component": ["認証モジュール", "データベース層", "APIゲートウェイ", "メッセージブローカー"],
        "feature": ["リアルタイム同期", "自動スケーリング", "障害回復", "データ暗号化"],
        "method": ["トークン化", "ベクトル検索", "強化学習", "勾配ブースティング"],
        "language": ["Python", "Rust", "TypeScript", "Go", "Kotlin", "Swift"],
        "task": ["非同期処理", "バッチジョブ", "データパイプライン", "マイグレーション"],
        "framework": ["FastAPI", "Next.js", "Axum", "Spring Boot", "Laravel"],
        "application": ["Webアプリ", "モバイルアプリ", "CLIツール", "マイクロサービス"],
        "field": ["自然言語処理", "コンピュータビジョン", "分散システム", "データベース"],
        "approach": ["Transformer", "拡散モデル", "強化学習", "GAN"],
        "system": ["レコメンド", "検索エンジン", "決済", "在庫管理"],
        "algorithm": ["クイックソート", "二分探索", "ダイクストラ法", "A*探索"],
        "complexity": ["n log n", "n²", "log n", "2ⁿ"],
    }

    lines = []
    for _ in range(num_lines):
        tmpl = random.choice(templates)
        for key, values in words.items():
            if "{" + key + "}" in tmpl:
                tmpl = tmpl.replace("{" + key + "}", random.choice(values), 1)

        defaults = ["デフォルト", "標準", "基本", "一般", "主要", "最新"]
        for key in ["percent", "error_message", "database", "query", "tool",
                     "process", "topic", "trend", "industry", "location", "event",
                     "book", "concept", "person", "work", "year", "category",
                     "problem", "strategy", "skill", "company", "product", "market"]:
            tmpl = tmpl.replace("{" + key + "}", random.choice(defaults), 1)
        lines.append(tmpl)

    return lines


def generate_code_corpus(num_lines: int = 60000) -> List[str]:
    """Code snippets in multiple languages."""
    snippets = [
        # Python
        "def {name}({args}) -> {ret_type}:\n    \"\"\"{docstring}\"\"\"\n    {body}\n    return {ret_val}",
        "class {class_name}:\n    def __init__(self, {init_args}):\n        self.{attr} = {attr}\n\n    def {method}(self, {args}):\n        {body}",
        "async def {name}({args}) -> {ret_type}:\n    {body}\n    await {await_expr}",
        "with open({filename}) as f:\n    data = json.load(f)\n    {body}",
        "try:\n    {body}\nexcept {exc_type} as e:\n    logger.error(f\"{msg}: {e}\")",
        "@{decorator}\ndef {name}({args}) -> {ret_type}:\n    {body}",

        # JavaScript/TypeScript
        "const {name} = ({args}: {types}) => {\n  {body}\n  return {ret_val};\n};",
        "interface {interface_name} {\n  {field}: {type};\n  {field2}: {type2};\n}",
        "export async function {name}({args}: {types}): Promise<{ret_type}> {\n  {body}\n}",
        "const [{state}, set{State}] = useState<{type}>({initial});",

        # Rust
        "fn {name}({args}: {types}) -> {ret_type} {\n    {body}\n}",
        "impl {struct_name} {\n    pub fn {method}(&self, {args}: {types}) -> {ret_type} {\n        {body}\n    }\n}",
        "let {var} = match {expr} {\n    Ok(val) => val,\n    Err(e) => return Err(e),\n};",

        # Shell
        "#!/bin/bash\nset -euo pipefail\n\n{body}",
        "for {var} in {iterable}; do\n    {body}\ndone",
        "find {path} -name \"{pattern}\" -exec {cmd} {} \\;",

        # SQL
        "SELECT {cols} FROM {table} WHERE {condition} ORDER BY {order} LIMIT {limit};",
        "INSERT INTO {table} ({cols}) VALUES ({values}) ON CONFLICT ({key}) DO UPDATE SET {updates};",
        "CREATE INDEX IF NOT EXISTS {idx_name} ON {table} ({cols});",
    ]

    words = {
        "name": ["process_data", "validate_input", "compute_score", "fetch_results",
                 "handle_request", "transform", "aggregate", "filter_items"],
        "args": ["x", "data", "items", "config", "params", "request"],
        "ret_type": ["int", "str", "bool", "Optional[dict]", "List[Item]", "Result<T>"],
        "docstring": ["Process the given data and return results.", "Validate input parameters.",
                      "Compute the weighted score.", "Fetch results from the API."],
        "body": ["result = process(x)", "return [item for item in items if condition]",
                 "await client.post(url, json=data)", "let filtered = items.filter(i => i.active)"],
        "ret_val": ["result", "True", "filtered", "response.json()", "Ok(data)"],
    }

    lines = []
    for _ in range(num_lines):
        tmpl = random.choice(snippets)
        for key, values in words.items():
            if "{" + key + "}" in tmpl:
                tmpl = tmpl.replace("{" + key + "}", random.choice(values), 1)
        lines.append(tmpl)

    return lines


# ═══════════════════════════════════════════════════════
# Tokenizer Creation
# ═══════════════════════════════════════════════════════

def create_tokenizer(
    vocab_size: int = 72000,
    output_dir: str = "tokenizer_small",
    en_lines: int = 80000,
    ja_lines: int = 60000,
    code_lines: int = 60000,
):
    """Create the specialized tokenizer."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

    print(f"🔤 Creating Small Model Tokenizer (vocab={vocab_size})")
    print("=" * 60)

    # ── Generate corpus ──
    print("📝 Generating diverse corpus...")
    random.seed(42)
    total_lines = en_lines + ja_lines + code_lines
    corpus = generate_diverse_corpus(total_lines)
    random.shuffle(corpus)
    corpus = corpus[:total_lines]
    print(f"   Total lines: {len(corpus):,}")

    # Write temp corpus
    corpus_path = Path(output_dir) / "_corpus_temp.txt"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text("\n".join(corpus), encoding="utf-8")
    print(f"   Corpus saved: {corpus_path} ({corpus_path.stat().st_size / 1024:.0f} KB)")

    # ── Create BPE tokenizer ──
    print("🔧 Training BPE tokenizer...")

    # Special tokens (same structure as nano, extended)
    special_tokens = [
        "<|bos|>",           # 0
        "<|eos|>",           # 1
        "<|pad|>",           # 2
        "<|unk|>",           # 3
        # FIM (Fill-in-the-Middle) for code
        "<|fim_prefix|>",    # 4
        "<|fim_suffix|>",    # 5
        "<|fim_middle|>",    # 6
        "<|fim_pad|>",       # 7
        "<|fim_hole|>",      # 8
        # Chat markers
        "<|system|>",        # 9
        "<|user|>",          # 10
        "<|assistant|>",     # 11
        # Tool use
        "<|tool_call|>",     # 12
        "<|tool_call_end|>", # 13
        "<|tool_resp|>",     # 14
        "<|tool_resp_end|>", # 15
        # Language markers
        "<|ja|>",            # 16
        "<|en|>",            # 17
        "<|code|>",          # 18
        # Code-specific
        "<|python|>",        # 19
        "<|javascript|>",    # 20
        "<|typescript|>",    # 21
        "<|rust|>",          # 22
        "<|shell|>",         # 23
        "<|sql|>",           # 24
        # Scratchpad
        "<|scratchpad|>",    # 25
        "<|scratchpad_end|>",# 26
    ]

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=1,  # Lower to capture more diversity
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train([str(corpus_path)], trainer)

    # Post-processor: add BOS/EOS
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<|bos|> $A <|eos|>",
        pair="<|bos|> $A <|eos|> $B:1 <|eos|>:1",
        special_tokens=[
            ("<|bos|>", tokenizer.token_to_id("<|bos|>")),
            ("<|eos|>", tokenizer.token_to_id("<|eos|>")),
        ],
    )

    # ── Save ──
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # HuggingFace format
    tokenizer.save(str(out / "tokenizer.json"))

    # Config
    config = {
        "add_prefix_space": False,
        "model_type": "bpe",
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<|unk|>",
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "pad_token": "<|pad|>",
        "special_tokens": special_tokens,
        "vocab_size": tokenizer.get_vocab_size(),
        "languages": ["en", "ja", "code"],
        "code_languages": ["python", "javascript", "typescript", "rust", "shell", "sql"],
    }
    with open(out / "tokenizer_config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Special tokens map (HF compatible)
    special_map = {
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "unk_token": "<|unk|>",
        "pad_token": "<|pad|>",
        "additional_special_tokens": [
            t for t in special_tokens
            if t not in ("<|bos|>", "<|eos|>", "<|unk|>", "<|pad|>")
        ],
    }
    with open(out / "special_tokens_map.json", "w") as f:
        json.dump(special_map, f, indent=2, ensure_ascii=False)

    # Cleanup temp corpus
    corpus_path.unlink()

    # ── Stats ──
    vocab = tokenizer.get_vocab_size()
    print(f"\n✅ Tokenizer created successfully!")
    print(f"   Vocab size: {vocab:,} (target: {vocab_size:,})")
    print(f"   Special tokens: {len(special_tokens)}")
    print(f"   Output: {out.absolute()}")
    print(f"   Corpus: {len(corpus):,} lines (EN:{en_lines:,} JA:{ja_lines:,} Code:{code_lines:,})")

    # Quick test
    test_texts = [
        "def hello_world() -> str:\n    return 'Hello, World!'",
        "Pythonの非同期処理について詳しく説明してください。",
        "impl Service for MyApp {\n    fn handle(&self) -> Result<()> {\n        Ok(())\n    }\n}",
        "async function fetchData(url: string): Promise<Response> { return await fetch(url); }",
    ]
    print("\n🧪 Encoding tests:")
    for text in test_texts:
        encoded = tokenizer.encode(text)
        print(f"   {len(encoded.ids):4d} tokens | {text[:60]}...")

    return tokenizer


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Create Small Model Tokenizer (EN+JA+Code)")
    parser.add_argument("--vocab", type=int, default=72000, help="Vocab size (65000-80000)")
    parser.add_argument("--output", default="tokenizer_small", help="Output directory")
    parser.add_argument("--en-lines", type=int, default=80000)
    parser.add_argument("--ja-lines", type=int, default=60000)
    parser.add_argument("--code-lines", type=int, default=60000)
    args = parser.parse_args()

    args.vocab = max(65000, min(80000, args.vocab))

    create_tokenizer(
        vocab_size=args.vocab,
        output_dir=args.output,
        en_lines=args.en_lines,
        ja_lines=args.ja_lines,
        code_lines=args.code_lines,
    )


if __name__ == "__main__":
    main()
