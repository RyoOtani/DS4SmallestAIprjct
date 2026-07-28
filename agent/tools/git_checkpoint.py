"""
git_checkpoint.py — Production-grade Git checkpoint / rollback system.

Features:
  - checkpoint(description) → create a git commit as restore point
  - rollback(checkpoint_id) → revert to any checkpoint
  - diff_since(checkpoint_id) → unified diff of changes
  - list_checkpoints() → enumerate all restore points
  - safe_mode: auto-stash before checkpoint, pop on rollback

Used by self-repair loop to safely experiment with code changes.
"""
import subprocess, os, json, time, uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class GitCheckpoint:
    """Git-based checkpoint system for safe code experimentation."""

    def __init__(self, repo_path: str = "."):
        self.repo = Path(repo_path).resolve()
        if not (self.repo / ".git").exists():
            raise ValueError(f"Not a git repository: {self.repo}")
        self._checkpoints: Dict[str, dict] = {}
        self._load_checkpoints()

    # ── Public API ──────────────────────────────────────────

    def checkpoint(self, description: str, auto_stash: bool = True) -> str:
        """
        Create a restore point. Returns checkpoint_id (git commit hash).
        Auto-stashes uncommitted changes before checkpointing.
        """
        if auto_stash and self._has_changes():
            subprocess.run(["git", "stash", "push", "-m", f"checkpoint-stash-{description[:40]}"],
                           cwd=self.repo, capture_output=True)

        # Stage everything
        subprocess.run(["git", "add", "-A"], cwd=self.repo, capture_output=True)

        # Create checkpoint commit
        result = subprocess.run(
            ["git", "commit", "--allow-empty", "-m", f"🔖 CHECKPOINT: {description}"],
            cwd=self.repo, capture_output=True, text=True
        )

        # Extract commit hash
        hash_result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=self.repo, capture_output=True, text=True
        )
        commit_hash = hash_result.stdout.strip()

        # Record
        self._checkpoints[commit_hash] = {
            "hash": commit_hash,
            "description": description,
            "timestamp": time.time(),
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        }
        self._save_checkpoints()

        # Pop stash if we stashed
        if auto_stash:
            stash_list = subprocess.run(
                ["git", "stash", "list"], cwd=self.repo, capture_output=True, text=True
            )
            if f"checkpoint-stash-{description[:40]}" in stash_list.stdout:
                subprocess.run(["git", "stash", "pop"], cwd=self.repo, capture_output=True)

        return commit_hash

    def rollback(self, checkpoint_id: str, hard: bool = True) -> bool:
        """
        Rollback to a checkpoint. If hard=True, discards all changes since checkpoint.
        Returns True on success.
        """
        if checkpoint_id not in self._checkpoints:
            # Try to find by prefix
            matches = [k for k in self._checkpoints if k.startswith(checkpoint_id)]
            if not matches:
                return False
            checkpoint_id = matches[0]

        # Stash current work before rolling back
        if self._has_changes():
            subprocess.run(["git", "stash", "push", "-m", "pre-rollback-safety"],
                           cwd=self.repo, capture_output=True)

        mode = "--hard" if hard else "--soft"
        result = subprocess.run(
            ["git", "reset", mode, checkpoint_id],
            cwd=self.repo, capture_output=True, text=True
        )
        return result.returncode == 0

    def diff_since(self, checkpoint_id: str, paths: List[str] = None) -> str:
        """Get unified diff between checkpoint and current state."""
        cmd = ["git", "diff", checkpoint_id]
        if paths:
            cmd.extend(["--"] + paths)
        result = subprocess.run(cmd, cwd=self.repo, capture_output=True, text=True)
        return result.stdout

    def diff_checkpoint_to_checkpoint(self, from_id: str, to_id: str,
                                       paths: List[str] = None) -> str:
        """Unified diff between two checkpoints."""
        cmd = ["git", "diff", from_id, to_id]
        if paths:
            cmd.extend(["--"] + paths)
        result = subprocess.run(cmd, cwd=self.repo, capture_output=True, text=True)
        return result.stdout

    def list_checkpoints(self, limit: int = 20) -> List[dict]:
        """List recent checkpoints."""
        result = subprocess.run(
            ["git", "log", "--oneline", f"-{limit}", "--grep=CHECKPOINT:", "--format=%H|%s|%ai"],
            cwd=self.repo, capture_output=True, text=True
        )
        checkpoints = []
        for line in result.stdout.strip().split("\n"):
            if not line: continue
            parts = line.split("|", 2)
            if len(parts) >= 2:
                checkpoints.append({
                    "hash": parts[0],
                    "short": parts[0][:8],
                    "description": parts[1].replace("🔖 CHECKPOINT: ", ""),
                    "time": parts[2] if len(parts) > 2 else "",
                })
        return checkpoints

    def current_hash(self) -> str:
        """Get current HEAD commit hash."""
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=self.repo, capture_output=True, text=True
        )
        return result.stdout.strip()

    def changed_files(self, since_checkpoint: str = None) -> List[str]:
        """List files changed since a checkpoint (or since last commit)."""
        cmd = ["git", "diff", "--name-only"]
        if since_checkpoint:
            cmd.append(since_checkpoint)
        else:
            cmd.append("HEAD")
        result = subprocess.run(cmd, cwd=self.repo, capture_output=True, text=True)
        return [f for f in result.stdout.strip().split("\n") if f]

    def apply_diff(self, diff_text: str, dry_run: bool = False) -> Tuple[bool, str]:
        """
        Apply a unified diff. Returns (success, output).
        If dry_run=True, only checks if the diff would apply cleanly.
        """
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
            f.write(diff_text)
            patch_path = f.name

        try:
            cmd = ["git", "apply"]
            if dry_run:
                cmd.append("--check")
            cmd.append(patch_path)
            result = subprocess.run(cmd, cwd=self.repo, capture_output=True, text=True)
            return result.returncode == 0, result.stderr or result.stdout
        finally:
            os.unlink(patch_path)

    # ── Internal ───────────────────────────────────────────

    def _has_changes(self) -> bool:
        result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=self.repo, capture_output=True, text=True
        )
        return bool(result.stdout.strip())

    def _save_checkpoints(self):
        cp_file = self.repo / ".tinyllm_checkpoints.json"
        # Keep only the 100 most recent
        sorted_cps = sorted(self._checkpoints.values(),
                            key=lambda x: x["timestamp"], reverse=True)[:100]
        with open(cp_file, 'w') as f:
            json.dump(sorted_cps, f, indent=2)

    def _load_checkpoints(self):
        cp_file = self.repo / ".tinyllm_checkpoints.json"
        if cp_file.exists():
            try:
                with open(cp_file) as f:
                    data = json.load(f)
                self._checkpoints = {cp["hash"]: cp for cp in data}
            except Exception:
                self._checkpoints = {}


# ── CLI ───────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    g = GitCheckpoint()

    if len(sys.argv) < 2:
        print("Usage: git_checkpoint.py <checkpoint|rollback|list|diff> [...]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "checkpoint":
        desc = sys.argv[2] if len(sys.argv) > 2 else "manual checkpoint"
        cid = g.checkpoint(desc)
        print(f"✅ Checkpoint created: {cid[:8]} — {desc}")

    elif cmd == "rollback":
        if len(sys.argv) < 3:
            print("Usage: rollback <checkpoint_id>")
            sys.exit(1)
        ok = g.rollback(sys.argv[2])
        print(f"{'✅ Rolled back' if ok else '❌ Failed'} to {sys.argv[2][:8]}")

    elif cmd == "list":
        for cp in g.list_checkpoints():
            print(f"  {cp['short']} — {cp['description']} ({cp['time']})")

    elif cmd == "diff":
        since = sys.argv[2] if len(sys.argv) > 2 else "HEAD~1"
        print(g.diff_since(since))

    elif cmd == "changed":
        since = sys.argv[2] if len(sys.argv) > 2 else None
        for f in g.changed_files(since):
            print(f"  {f}")
