# TinyLLM v2 Phase 2

Phase 2 adds the reliability layer for autonomous software engineering.

## Pipeline

`Repository Index -> Code RAG -> Agent -> Build/Test -> Critic -> Evaluator -> Repair`

### New components

- `code_index/code_graph.py`: hybrid repository retrieval and lightweight dependency graph.
- `python/coding_agent/checkpoint.py`: Git checkpoints and rollback.
- `python/coding_agent/review.py`: independent Critic + Evaluator pipeline.
- `python/coding_agent/phase2.py`: orchestration entry point.
- `Toolset.code_context`: retrieves relevant code before edits.

## Recommended workflow

1. Index the repository.
2. Create a Git checkpoint.
3. Let the agent inspect and modify code.
4. Run tests/build.
5. Critic reviews the diff.
6. Evaluator decides pass/fail.
7. If failed, feed the review back to the agent for repair.
8. Roll back when a change is unsafe.

## Example

```bash
python -m code_index.indexer ./workspace -o ./workspace/code_index/index.jsonl
python -m code_index.code_graph
python -m python.coding_agent.phase2 ./workspace "Fix the failing authentication tests"
```

The Phase 2 pipeline is intentionally bounded. It should not be treated as a fully autonomous production coding system until sandboxing, permissions, audit logs, and stronger test isolation are enabled.
