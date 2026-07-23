# TinyLLM Phase 4 — AI Software Engineer Integration

Phase 4 connects the multi-agent orchestration direction to concrete runtime components:

- OpenAI-compatible LLM provider
- Workspace-scoped file/search/command tools
- Persistent agent memory
- Git checkpoint/rollback primitive
- Inspection → Verification → Critical Review cycle

## Run with a compatible local or remote model

```bash
export TINYLLM_BASE_URL=http://localhost:8000/v1
export TINYLLM_MODEL=your-model
python tinyllm_engineer.py "Inspect this repository and identify the highest-risk bug"
```

## Safe mock mode

```bash
python tinyllm_engineer.py "Inspect this repository" --mock
```

## Important limitation

This phase intentionally does not pretend to be a fully autonomous frontier coding agent. The provider and tool primitives are now connected, but robust tool-call schemas, model-driven edit execution, sandbox isolation, and iterative repair policies still need production hardening.

## Recommended Phase 5

1. Structured tool calling with JSON schema.
2. Model-driven file edits with patch validation.
3. Code-RAG context injection into every agent role.
4. Build/test failure parsing.
5. Critic → Evaluator → Fixer loop with bounded retries.
6. Git checkpoint before edits and automatic rollback.
7. Docker/Firecracker sandbox.
8. Human approval gates for destructive operations.
