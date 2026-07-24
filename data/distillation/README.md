# TinyLLM Distillation Dataset

**GitHub**: https://github.com/RyoOtani/DS4SmallestAIprjctDistilldata-DeepSeek-v4Flash-Pro-  
**Teacher**: DeepSeek-V4-Flash-Pro  
**Student**: TinyLLM-nano / small  
**Format**: JSONL (one JSON object per line)  
**Fields**: `text` — the text content for distillation

## ファイル構成

| File | Description | Lines | Size |
|------|-------------|-------|------|
| `sample_code_distill.jsonl` | コード生成サンプル (Python/C/JS/Rust) | 500 | ~250 KB |
| `sample_instruction_distill.jsonl` | 命令応答サンプル (日英バイリンガル) | 200 | ~150 KB |
| `sample_fim_distill.jsonl` | Fill-in-the-Middle サンプル | 300 | ~200 KB |
| `generate_large_dataset.py` | 大規模データ生成スクリプト | — | — |

## データ形式

各行は以下の JSON フォーマット:

```json
{"text": "Write Python code to implement a binary search tree..."}
```

`distill.py` が自動的に `text` フィールドを読み込み、必要に応じて FIM (Fill-in-the-Middle) フォーマットに変換します。

## 使い方

```bash
# 最小限のサンプルデータで蒸留テスト
python python/train/distill.py \
  --teacher deepseek-ai/DeepSeek-V3 \
  --student Qwen/Qwen3-5-9B \
  --data data/distillation/sample_code_distill.jsonl \
  --output output/distilled_test \
  --max-steps 100

# 大規模データを生成してから本番蒸留
python data/distillation/generate_large_dataset.py \
  --output data/distillation/large_dataset.jsonl \
  --n-samples 10000

python python/train/distill.py \
  --data data/distillation/large_dataset.jsonl \
  --max-steps 10000
```

## データソース

サンプルデータは主に以下のソースから抽出:
- 公開コードリポジトリ (Python, C, JavaScript, Rust, Go)
- 技術文書・チュートリアル
- 命令応答ペア

> **Note**: This is a sample dataset for benchmarking. For production training,
> use `generate_large_dataset.py` to create a larger dataset from your teacher model.
