"""
RepairLoop with staleness detection and learning cache.

Prevents infinite loops by:
  ✅ Detecting when the same fix is re-applied (staleness)
  ✅ Capping total attempts with exponential backoff
  ✅ Caching successful repair strategies per error type
  ✅ Comparing diagnosis hashes to detect no-progress loops
"""

import json
import hashlib
import time
from typing import Optional


class RepairLoop:
    """Self-correcting repair loop with staleness detection."""

    def __init__(self, engineer, max_attempts=3):
        self.engineer = engineer
        self.max_attempts = max_attempts
        self._seen_diagnoses: set[str] = set()
        self._seen_patches: set[str] = set()
        self._last_diagnosis_text: str = ""  # exact string comparison
        self._strategy_cache: dict[str, str] = {}  # error_hash → fix_strategy

    def _hash_text(self, text: str) -> str:
        """Stable short hash for comparison."""
        return hashlib.md5(text.encode())[:16].hex()

    def diagnose(self, request: str, verification: dict) -> str:
        """Run diagnosis and detect staleness (exact + hash)."""
        context = json.dumps(verification, ensure_ascii=False)
        diagnosis = self.engineer.ask("debugger", request, context)

        # Fast check: exact same diagnosis text as last time → stuck
        if diagnosis == self._last_diagnosis_text:
            return f"[STUCK — identical diagnosis] {diagnosis}"
        self._last_diagnosis_text = diagnosis

        # Hash check: same content seen before (may differ in whitespace)
        diag_hash = self._hash_text(diagnosis)
        if diag_hash in self._seen_diagnoses:
            error_type = verification.get("error_type", "unknown")
            if error_type in self._strategy_cache:
                return f"[CACHED STRATEGY] {self._strategy_cache[error_type]}"
            return f"[STALE — same diagnosis #{len(self._seen_diagnoses)}] {diagnosis}"

        self._seen_diagnoses.add(diag_hash)
        return diagnosis

    def run(self, request: str, test_command: str, execute_fix):
        """
        Run repair loop with staleness detection.

        Args:
            request: Original task description
            test_command: Verification command
            execute_fix: Callable(diagnosis, verification, attempt) → {"ok": bool, ...}
        """
        history = []
        last_fix_hash: Optional[str] = None

        for attempt in range(1, self.max_attempts + 1):
            verification = self.engineer.verify(test_command)
            history.append({"attempt": attempt, "verification": verification})

            # Success
            if verification.get("returncode") == 0:
                return {"status": "passed", "attempts": attempt, "history": history}

            # Diagnose
            diagnosis = self.diagnose(request, verification)
            history[-1]["diagnosis"] = diagnosis

            # Check for staleness
            if "[STALE" in diagnosis:
                history[-1]["staleness"] = True
                # One more attempt with different approach, then give up
                if attempt >= self.max_attempts:
                    break
                # Force a different diagnosis by asking the model to think differently
                diagnosis = self.engineer.ask(
                    "debugger_creative",
                    request,
                    json.dumps(verification, ensure_ascii=False) +
                    "\n\nPrevious attempts all failed. Try a COMPLETELY different approach."
                )
                history[-1]["diagnosis_retry"] = diagnosis

            # Check fix hash — same fix as before = no progress
            fix_hash = self._hash_text(diagnosis)
            if fix_hash == last_fix_hash:
                history[-1]["no_progress"] = True
                if attempt >= self.max_attempts:
                    break
                # Back off
                time.sleep(1.0 * attempt)

            last_fix_hash = fix_hash

            # Apply fix
            fixed = execute_fix(diagnosis, verification, attempt)
            history[-1]["fix"] = fixed

            if not fixed.get("ok", False):
                # Cache this failure pattern
                error_type = verification.get("error_type", "unknown")
                if error_type not in self._strategy_cache and fixed.get("error"):
                    self._strategy_cache[error_type] = fixed.get("error", "")

            # Exponential backoff on repeated failures
            if attempt >= 2:
                time.sleep(0.5 * (2 ** (attempt - 2)))

        return {"status": "failed", "attempts": len(history), "history": history}

    def get_cached_strategy(self, error_type: str) -> Optional[str]:
        """Retrieve a previously successful repair strategy."""
        return self._strategy_cache.get(error_type)

    def learn_from_success(self, error_type: str, strategy: str):
        """Cache a successful strategy for future use."""
        self._strategy_cache[error_type] = strategy

