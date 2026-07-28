"""
self_repair.py — Autonomous self-repair loop using checkpoint/sandbox/diff tools.

Flow:
  1. CHECKPOINT  — save current state to git
  2. DIAGNOSE    — analyze failure (test output, error messages)
  3. PLAN        — generate fix plan
  4. EDIT        — apply code changes via diff
  5. TEST        — run tests in sandbox
  6. VERIFY      — if pass → keep; if fail → rollback + retry (up to N times)
  7. COMMIT      — if all good, commit the fix

Integrates:
  - GitCheckpoint  for safe experimentation
  - CodeSandbox    for isolated test execution
  - DiffEditor     for precise code modifications
"""
import os, sys, time, json, re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable

# Add parent (agent/) to path for imports
_AGENT_DIR = Path(__file__).resolve().parent.parent
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from tools.git_checkpoint import GitCheckpoint
from tools.sandbox import CodeSandbox, SandboxResult
from tools.diff_editor import DiffEditor


class RepairAttempt:
    """Record of a single repair attempt."""
    def __init__(self, attempt_num: int):
        self.num = attempt_num
        self.checkpoint_id: str = ""
        self.diagnosis: str = ""
        self.plan: str = ""
        self.diff_text: str = ""
        self.test_result: Optional[SandboxResult] = None
        self.success: bool = False
        self.runtime_ms: float = 0

    def to_dict(self) -> dict:
        return {
            "attempt": self.num,
            "checkpoint": self.checkpoint_id[:8] if self.checkpoint_id else "",
            "diagnosis": self.diagnosis[:200],
            "plan": self.plan[:200],
            "success": self.success,
            "test_result": self.test_result.to_dict() if self.test_result else None,
            "runtime_ms": round(self.runtime_ms, 1),
        }


class SelfRepairLoop:
    """
    Autonomous code repair system.

    repair_fn is a callback that receives (diagnosis, files, test_output)
    and returns (plan_text, diff_text). This is where LLM integration happens.
    """

    MAX_ATTEMPTS = 5

    def __init__(self, repo_path: str = ".",
                 repair_fn: Callable = None,
                 test_command: str = None,
                 test_file: str = None):
        self.repo = Path(repo_path).resolve()
        self.checkpoint = GitCheckpoint(str(self.repo))
        self.diff_editor = DiffEditor(str(self.repo))
        self.repair_fn = repair_fn or self._default_repair_fn
        self.test_command = test_command
        self.test_file = test_file
        self.attempts: List[RepairAttempt] = []

    # ── Main Loop ─────────────────────────────────────────

    def repair(self, error_description: str = "",
               changed_files: List[str] = None,
               max_attempts: int = None) -> RepairAttempt:
        """
        Run the full self-repair loop.
        Returns the final (successful) RepairAttempt, or the last attempt.
        """
        max_attempts = max_attempts or self.MAX_ATTEMPTS
        max_attempts = min(max_attempts, self.MAX_ATTEMPTS)

        # ── Step 0: Create initial checkpoint ──
        cp_id = self.checkpoint.checkpoint(
            f"self-repair-start: {error_description[:60]}"
        )

        for attempt_num in range(1, max_attempts + 1):
            print(f"\n🔧 Repair attempt {attempt_num}/{max_attempts}")
            att = RepairAttempt(attempt_num)
            att.checkpoint_id = cp_id
            t0 = time.time()

            # ── Step 1: Diagnose ──
            att.diagnosis = self._diagnose(error_description, changed_files)

            # ── Step 2: Plan fix ──
            affected_files = changed_files or self.checkpoint.changed_files(cp_id)
            test_output = self._get_test_output()

            att.plan, att.diff_text = self.repair_fn(
                diagnosis=att.diagnosis,
                affected_files=affected_files,
                test_output=test_output,
                attempt=attempt_num,
            )

            if not att.diff_text or not att.diff_text.strip():
                print("  ⚠️  No diff generated — skipping")
                att.success = False
                self.attempts.append(att)
                continue

            # ── Step 3: Validate diff ──
            valid, msg = self.diff_editor.validate_diff(att.diff_text)
            if not valid:
                print(f"  ❌ Invalid diff: {msg[:100]}")
                att.diagnosis += f"\nDiff validation failed: {msg}"
                self.attempts.append(att)
                continue

            # ── Step 4: Apply edit ──
            ok, msg = self.diff_editor.apply_diff(att.diff_text)
            if not ok:
                print(f"  ❌ Diff apply failed: {msg[:100]}")
                self.checkpoint.rollback(cp_id, hard=True)
                att.diagnosis += f"\nDiff apply failed: {msg}"
                self.attempts.append(att)
                continue

            # ── Step 5: Run tests ──
            att.test_result = self._run_tests()

            # ── Step 6: Verify ──
            if att.test_result and att.test_result.success:
                att.success = True
                att.runtime_ms = (time.time() - t0) * 1000
                self.attempts.append(att)
                print(f"  ✅ Repair successful! (attempt {attempt_num}, {att.runtime_ms:.0f}ms)")
                # Commit the fix
                self.checkpoint.checkpoint(
                    f"self-repair-success: {error_description[:60]} (attempt {attempt_num})"
                )
                return att
            else:
                # Rollback and retry
                print(f"  ❌ Test failed — rolling back")
                self.checkpoint.rollback(cp_id, hard=True)
                # Enrich diagnosis for next attempt
                if att.test_result:
                    error_description += f"\n[Attempt {attempt_num}] Tests failed:\n{att.test_result.stderr[:300]}\n{att.test_result.stdout[:300]}"
                att.runtime_ms = (time.time() - t0) * 1000
                self.attempts.append(att)

        # All attempts exhausted
        print(f"\n⚠️  All {max_attempts} repair attempts exhausted. Rolling back.")
        self.checkpoint.rollback(cp_id, hard=True)
        return self.attempts[-1] if self.attempts else RepairAttempt(0)

    # ── Quick Fix (single attempt, no loop) ───────────────

    def quick_fix(self, diff_text: str, test_cmd: str = None) -> bool:
        """Apply a single diff and test. Rollback on failure."""
        cp_id = self.checkpoint.checkpoint("quick-fix")

        valid, msg = self.diff_editor.validate_diff(diff_text)
        if not valid:
            self.checkpoint.rollback(cp_id, hard=True)
            return False

        ok, _ = self.diff_editor.apply_diff(diff_text)
        if not ok:
            self.checkpoint.rollback(cp_id, hard=True)
            return False

        result = self._run_tests(test_cmd)
        if result and result.success:
            self.checkpoint.checkpoint("quick-fix-success")
            return True

        self.checkpoint.rollback(cp_id, hard=True)
        return False

    # ── Report ────────────────────────────────────────────

    def report(self) -> str:
        """Generate a summary report of all repair attempts."""
        lines = ["# Self-Repair Report", f"Total attempts: {len(self.attempts)}", ""]
        for att in self.attempts:
            status = "✅" if att.success else "❌"
            lines.append(f"## Attempt {att.num} {status}")
            lines.append(f"- Checkpoint: `{att.checkpoint_id[:8]}`")
            lines.append(f"- Diagnosis: {att.diagnosis[:150]}")
            lines.append(f"- Plan: {att.plan[:150]}")
            if att.test_result:
                lines.append(f"- Test: exit={att.test_result.exit_code} time={att.test_result.runtime_ms:.0f}ms")
            lines.append(f"- Runtime: {att.runtime_ms:.0f}ms")
            lines.append("")
        return '\n'.join(lines)

    # ── Internals ─────────────────────────────────────────

    def _diagnose(self, error_description: str, changed_files: List[str] = None) -> str:
        """Build a diagnosis from error description and changed files."""
        parts = [f"Error: {error_description}" if error_description else "No error description provided"]

        if changed_files:
            parts.append(f"\nChanged files ({len(changed_files)}):")
            for f in changed_files[:10]:
                parts.append(f"  - {f}")

        # Try to extract structured info
        if "SyntaxError" in error_description or "syntax error" in error_description.lower():
            parts.append("\nType: Syntax error — likely a typo or missing punctuation")
        elif "ImportError" in error_description or "ModuleNotFoundError" in error_description:
            parts.append("\nType: Import error — check dependencies or import paths")
        elif "AssertionError" in error_description or "assert" in error_description.lower():
            parts.append("\nType: Assertion failure — logic or expected value mismatch")
        elif "segfault" in error_description.lower() or "SIGSEGV" in error_description:
            parts.append("\nType: Memory error — likely buffer overflow or null pointer")

        return '\n'.join(parts)

    def _run_tests(self, cmd: str = None) -> Optional[SandboxResult]:
        """Run the configured test command in sandbox."""
        command = cmd or self.test_command
        if not command:
            return None

        sandbox = CodeSandbox(str(self.repo), timeout=60)
        try:
            if self.test_file:
                return sandbox.run_python_file(self.test_file)
            else:
                return sandbox.run_shell(command,
                    allowlist=["python", "python3", "pytest", "make", "gcc", "cc"])
        finally:
            sandbox.cleanup()

    def _get_test_output(self) -> str:
        """Get current test output for diagnosis."""
        result = self._run_tests()
        if result:
            return f"exit={result.exit_code}\nstdout:\n{result.stdout[:500]}\nstderr:\n{result.stderr[:500]}"
        return "(no test configured)"

    @staticmethod
    def _default_repair_fn(diagnosis: str, affected_files: List[str],
                           test_output: str, attempt: int) -> Tuple[str, str]:
        """
        Default repair function (no LLM — template-based).
        Replace with LLM-powered repair for production use.
        """
        plan = f"Auto-repair attempt {attempt}\nDiagnosis: {diagnosis[:200]}"
        diff = ""  # No diff = skip (LLM would generate this)
        return plan, diff


# ── LLM Repair Function Factory ──────────────────────────

def make_llm_repair_fn(provider):
    """
    Create a repair function powered by an LLM provider.
    Usage:
        repair_fn = make_llm_repair_fn(my_provider)
        loop = SelfRepairLoop(repair_fn=repair_fn)
    """
    def llm_repair(diagnosis: str, affected_files: List[str],
                   test_output: str, attempt: int) -> Tuple[str, str]:
        prompt = f"""You are an expert code repair system. Diagnose and fix the following error.

## Diagnosis
{diagnosis}

## Affected Files
{chr(10).join(f'- {f}' for f in affected_files) if affected_files else '(unknown)'}

## Test Output
{test_output[:1000]}

## Instructions
1. Analyze the error and identify the root cause
2. Generate a unified diff (git format) that fixes the issue
3. The diff must be minimal — only change what's necessary
4. Output format:

PLAN: <one-line fix description>
DIFF:
```diff
<unified diff here>
```"""

        try:
            response, _ = provider.chat(prompt)
        except Exception as e:
            return f"LLM error: {e}", ""

        # Parse response
        plan = ""
        diff = ""
        plan_match = re.search(r'PLAN:\s*(.+?)(?:\n|$)', response)
        if plan_match:
            plan = plan_match.group(1).strip()

        diff_match = re.search(r'```diff\n(.+?)```', response, re.DOTALL)
        if not diff_match:
            diff_match = re.search(r'```\n(---.+?)```', response, re.DOTALL)
        if diff_match:
            diff = diff_match.group(1).strip()

        return plan, diff

    return llm_repair


# ── CLI ───────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: self_repair.py <error_description> [--test-cmd CMD] [--files f1 f2 ...]")
        sys.exit(1)

    error = sys.argv[1]
    test_cmd = None
    files = []

    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--test-cmd" and i + 1 < len(args):
            test_cmd = args[i + 1]; i += 2
        elif args[i] == "--files":
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                files.append(args[i]); i += 1
        else:
            i += 1

    loop = SelfRepairLoop(test_command=test_cmd)
    result = loop.repair(error_description=error, changed_files=files or None)

    if result.success:
        print(f"\n✅ Repaired in {result.num} attempt(s)")
    else:
        print(f"\n❌ Failed after {len(loop.attempts)} attempt(s)")

    print(loop.report())
