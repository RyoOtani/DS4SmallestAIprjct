"""
diff_editor.py — Production-grade unified diff generation and application.

Features:
  - diff_files(original, modified) → unified diff
  - apply_diff(diff_text, target_dir) → apply with conflict detection
  - smart_patch(target_file, diff_text) → intelligent merge with fuzzy matching
  - create_edit(prompt, files) → LSP-assisted edit generation (placeholder for LLM)
  - validate_diff(diff_text) → check if diff is well-formed
"""
import subprocess, os, re, difflib, tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict


class DiffEditor:
    """Unified diff generation, validation, and application."""

    def __init__(self, repo_root: str = "."):
        self.root = Path(repo_root).resolve()

    # ── Diff Generation ───────────────────────────────────

    def diff_files(self, original_path: str, modified_path: str,
                   context_lines: int = 3) -> str:
        """Generate unified diff between two files."""
        orig = Path(original_path)
        mod = Path(modified_path)

        if not orig.exists() or not mod.exists():
            return ""

        with open(orig) as f:
            orig_lines = f.readlines()
        with open(mod) as f:
            mod_lines = f.readlines()

        diff = difflib.unified_diff(
            orig_lines, mod_lines,
            fromfile=str(orig), tofile=str(mod),
            n=context_lines
        )
        return ''.join(diff)

    def diff_against_git(self, filepath: str, base: str = "HEAD") -> str:
        """Generate diff of a file against a git ref."""
        result = subprocess.run(
            ["git", "diff", base, "--", filepath],
            cwd=self.root, capture_output=True, text=True
        )
        return result.stdout

    def diff_working_tree(self, paths: List[str] = None) -> str:
        """Diff of all uncommitted changes."""
        cmd = ["git", "diff"]
        if paths:
            cmd.extend(["--"] + paths)
        result = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True)
        return result.stdout

    # ── Diff Application ──────────────────────────────────

    def apply_diff(self, diff_text: str, target_dir: str = None,
                   dry_run: bool = False) -> Tuple[bool, str]:
        """
        Apply a unified diff using git apply (handles most edge cases).
        Returns (success, output).
        """
        target = Path(target_dir or self.root)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
            f.write(diff_text)
            patch_path = f.name

        try:
            cmd = ["git", "apply"]
            if dry_run:
                cmd.append("--check")
            cmd.extend(["--directory", str(target)])
            cmd.append(patch_path)

            result = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True)
            return result.returncode == 0, result.stderr or result.stdout
        finally:
            os.unlink(patch_path)

    def apply_patch_file(self, patch_path: str, dry_run: bool = False) -> Tuple[bool, str]:
        """Apply a .patch file from disk."""
        cmd = ["git", "apply"]
        if dry_run:
            cmd.append("--check")
        cmd.append(patch_path)

        result = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True)
        return result.returncode == 0, result.stderr or result.stdout

    def smart_apply(self, diff_text: str, target_file: str,
                    fuzzy: bool = True) -> Tuple[bool, str]:
        """
        Intelligent diff application with fuzzy matching.
        Falls back to patch command with fuzz factor for near-miss hunks.
        """
        target = Path(target_file)
        if not target.exists():
            return False, f"Target file not found: {target_file}"

        # Try git apply first (strict)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
            f.write(diff_text)
            patch_path = f.name

        try:
            cmd = ["git", "apply", "--check", patch_path]
            result = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True)
            if result.returncode == 0:
                # Apply cleanly
                cmd = ["git", "apply", patch_path]
                result2 = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True)
                os.unlink(patch_path)
                return result2.returncode == 0, result2.stderr or result2.stdout

            # Strict failed — try fuzzy with patch command
            if fuzzy:
                cmd = ["patch", "-f", "-p0", "-i", patch_path, str(target)]
                result3 = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True)
                os.unlink(patch_path)
                return result3.returncode == 0, result3.stdout or result3.stderr

            os.unlink(patch_path)
            return False, result.stderr or result.stdout
        except Exception as e:
            try: os.unlink(patch_path)
            except: pass
            return False, str(e)

    # ── Diff Validation ───────────────────────────────────

    def validate_diff(self, diff_text: str) -> Tuple[bool, str]:
        """Check if a diff is well-formed and would apply cleanly."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
            f.write(diff_text)
            patch_path = f.name

        try:
            # --check validates without applying
            result = subprocess.run(
                ["git", "apply", "--check", patch_path],
                cwd=self.root, capture_output=True, text=True
            )
            if result.returncode == 0:
                return True, "Diff is valid and would apply cleanly"
            return False, result.stderr.strip()
        finally:
            os.unlink(patch_path)

    def parse_diff(self, diff_text: str) -> List[dict]:
        """Parse a unified diff into structured hunks."""
        hunks = []
        current_file = None
        current_hunk = None

        for line in diff_text.split('\n'):
            if line.startswith('--- '):
                current_file = line[4:].split('\t')[0]
            elif line.startswith('+++ '):
                current_file = line[4:].split('\t')[0]
            elif line.startswith('@@'):
                if current_hunk:
                    hunks.append(current_hunk)
                # Parse @@ -old_start,old_count +new_start,new_count @@
                m = re.match(r'@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@(.*)', line)
                if m:
                    current_hunk = {
                        "file": current_file,
                        "old_start": int(m.group(1)),
                        "old_count": int(m.group(2) or 1),
                        "new_start": int(m.group(3)),
                        "new_count": int(m.group(4) or 1),
                        "context": m.group(5).strip(),
                        "lines": [],
                    }
            elif current_hunk is not None:
                current_hunk["lines"].append(line)

        if current_hunk:
            hunks.append(current_hunk)
        return hunks

    # ── File Operations ───────────────────────────────────

    def create_edit(self, description: str, edits: List[dict]) -> str:
        """
        Create a structured edit description.
        edits: [{"file": path, "action": "replace|insert|delete",
                  "old": "...", "new": "...", "line": N}]
        Returns unified diff string if successful.
        """
        diffs = []
        for edit in edits:
            filepath = self.root / edit["file"]
            if not filepath.exists():
                diffs.append(f"# Cannot edit: {edit['file']} does not exist")
                continue

            content = filepath.read_text()
            lines = content.split('\n')

            if edit["action"] == "replace":
                old = edit.get("old", "")
                new = edit.get("new", "")
                if old in content:
                    new_content = content.replace(old, new, 1)
                    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.tmp', delete=False)
                    tmp.write(new_content)
                    tmp.close()
                    diff = self.diff_files(str(filepath), tmp.name)
                    os.unlink(tmp.name)
                    diffs.append(diff)
                else:
                    diffs.append(f"# Could not find text to replace in {edit['file']}")

            elif edit["action"] == "insert":
                line_no = edit.get("line", len(lines))
                new_text = edit.get("new", "")
                new_lines = lines[:line_no] + [new_text] + lines[line_no:]
                new_content = '\n'.join(new_lines)
                tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.tmp', delete=False)
                tmp.write(new_content)
                tmp.close()
                diff = self.diff_files(str(filepath), tmp.name)
                os.unlink(tmp.name)
                diffs.append(diff)

            elif edit["action"] == "delete":
                old = edit.get("old", "")
                if old in content:
                    new_content = content.replace(old, "", 1)
                    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.tmp', delete=False)
                    tmp.write(new_content)
                    tmp.close()
                    diff = self.diff_files(str(filepath), tmp.name)
                    os.unlink(tmp.name)
                    diffs.append(diff)

        return '\n'.join(d for d in diffs if d and not d.startswith('#'))


# ── CLI ───────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    d = DiffEditor()

    if len(sys.argv) < 2:
        print("Usage: diff_editor.py <diff|apply|validate|parse> [...]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "diff":
        if len(sys.argv) >= 4:
            print(d.diff_files(sys.argv[2], sys.argv[3]))
        else:
            print(d.diff_working_tree())

    elif cmd == "apply":
        diff_text = sys.stdin.read() if len(sys.argv) < 3 else open(sys.argv[2]).read()
        ok, msg = d.apply_diff(diff_text)
        print(f"{'✅ Applied' if ok else '❌ Failed'}: {msg}")

    elif cmd == "validate":
        diff_text = sys.stdin.read() if len(sys.argv) < 3 else open(sys.argv[2]).read()
        ok, msg = d.validate_diff(diff_text)
        print(f"{'✅ Valid' if ok else '❌ Invalid'}: {msg}")

    elif cmd == "parse":
        diff_text = sys.stdin.read() if len(sys.argv) < 3 else open(sys.argv[2]).read()
        for hunk in d.parse_diff(diff_text):
            print(f"  {hunk['file']}: @@ -{hunk['old_start']},{hunk['old_count']} +{hunk['new_start']},{hunk['new_count']} @@")
            for l in hunk['lines'][:3]:
                print(f"    {l}")
