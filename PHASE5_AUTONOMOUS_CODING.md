# TinyLLM Phase 5 — Autonomous Coding Loop

Phase 5 adds the first explicit end-to-end edit/verify/repair loop.

```text
Request
  ↓
Repository Inspection
  ↓
Plan
  ↓
Unified Diff Generation
  ↓
Patch Validation + Apply
  ↓
Build / Test
  ↓
Failure Diagnosis
  ↓
Repair Diff
  ↓
Re-test (bounded retries)
  ↓
Final Review
```

## Run

```bash
python tinyllm_autocode.py "Change the feature" \
  --workspace ./my-project \
  --base-url http://localhost:8000/v1 \
  --model your-model \
  --test-command "python -m pytest -q"
```

## Safety boundary

- File edits must be unified diffs.
- Patches are restricted to the workspace.
- Repair attempts are bounded.
- Verification runs are explicit.
- For production use, run commands inside a real sandbox/container and add human approval gates for destructive operations.

## Next

Phase 6 should add:
- structured JSON tool calling,
- AST-aware patch generation,
- Code RAG context injection,
- Git commit checkpoint + automatic rollback,
- Docker/Firecracker sandbox,
- multi-agent handoffs,
- stronger regression testing,
- explicit permission policies.
