# TinyLLM Phase 3 — Multi-Agent Coding System

Phase 3 adds a deterministic orchestration layer for a Planner → Architect → Coder → Tester → Critic → Evaluator → Debugger workflow.

## Architecture

```text
User Request
    |
 Planner
    |
 Architect
    |
 Coder
    |
 Build / Test
    |
 Critic
    |
 Evaluator
   / \
PASS FAIL
 |     |
Done  Debugger -> Repair cycle
```

## Important

This phase provides the orchestration and role contracts. It does **not** pretend that placeholder role logic is equivalent to a frontier model. Connect a real model through the existing provider abstraction, then connect the coder role to the Phase 2 tool layer.

## Run

```bash
python phase3_multi_agent.py "Add a feature to this repository"
```

## Test

```bash
python -m pytest -q
```

## Next integration steps

1. Connect Planner/Architect/Critic/Evaluator prompts to an actual LLM provider.
2. Connect Coder to the Phase 2 workspace tools.
3. Connect Tester to build/test commands.
4. Feed Code RAG results into Architect and Coder context.
5. Add Git checkpoint/rollback around each implementation cycle.
6. Persist agent memory between runs.
