# TinyLLM v2 — Runtime + Coding Agent

This branch evolves the original TinyLLM into two coordinated layers:

1. **TinyLLM Runtime** — keep the existing C inference core and expose a provider abstraction so local GGUF inference and stronger external/OpenAI-compatible models can be selected without rewriting the agent.
2. **Coding Agent** — repository-aware plan/act/observe loop with workspace-safe file access, search, command execution, tests, and bounded self-repair.

## Current v2 additions

- `python/runtime/provider.py`: model-provider abstraction
- `python/coding_agent/`: bounded coding agent and workspace-safe tools
- `code_index/indexer.py`: lightweight source index with symbols/imports
- `python/tools/sandbox_v2.py`: safer command execution helper

## Quick start

```bash
python -m code_index.indexer .
```

Configure an OpenAI-compatible local or remote endpoint:

```bash
export TINYLLM_PROVIDER=openai_compatible
export TINYLLM_BASE_URL=http://localhost:11434/v1
export TINYLLM_MODEL=qwen2.5-coder:7b
python -m python.coding_agent.cli "Inspect this project, fix the failing tests, and verify the build" --workspace .
```

Or use a CLI model:

```bash
export TINYLLM_PROVIDER=command
export TINYLLM_COMMAND='ollama run qwen2.5-coder:7b'
python -m python.coding_agent.cli "Add a unit test for the tokenizer" --workspace .
```

## Architecture

```text
                    TinyLLM v2
                         |
          +--------------+--------------+
          |                             |
    C Runtime                    Coding Agent
  GGUF / KV / MoE                Planner / Repair
  Quantization                   Tools / Memory
          |                             |
          +-------------+---------------+
                        |
                Provider Interface
                 /            \
          Local Runtime    Strong LLM API
                        |
                 Code Index / RAG
                        |
               Sandbox / Build / Test
```

## Important status

This is a **working v2 foundation**, not a claim that the repository already contains a DeepSeek-V4-class model implementation. The runtime and agent are separate concerns: the agent can use a stronger model through the provider interface while the C runtime continues to mature.

Next milestones:

- [ ] connect C runtime as a first-class provider
- [ ] AST-aware indexing with tree-sitter adapters
- [ ] dependency/call graph retrieval
- [ ] patch-based file editing instead of full-file replacement
- [ ] git checkpoint/rollback
- [ ] critic → evaluator → fixer loop
- [ ] containerized build/test execution
- [ ] multi-agent roles (architect/coder/tester/reviewer)
- [ ] benchmark suite for repository-level coding tasks
