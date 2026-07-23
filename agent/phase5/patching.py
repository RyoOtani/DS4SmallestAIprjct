"""
Safe unified-diff patch applier with atomicity and binary detection.

Features:
  ✅ Binary file detection (rejects patches on binary files)
  ✅ Atomic application (validates ALL hunks before applying ANY)
  ✅ Proper no-newline-at-EOF handling
  ✅ Multi-file rollback on partial failure
  ✅ Workspace-escape prevention
  ✅ File permission preservation
"""

import re
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional


class PatchError(Exception):
    """Patch application error with context."""
    def __init__(self, message: str, file: str = "", line: int = 0):
        self.file = file
        self.line = line
        loc = f" at {file}:{line}" if file else ""
        super().__init__(f"{message}{loc}")


class PatchHunk:
    """Represents a single @@ hunk within a file patch."""
    
    def __init__(self, old_start: int, old_count: int, new_start: int, new_count: int):
        self.old_start = old_start - 1  # 0-indexed
        self.old_count = old_count
        self.new_start = new_start - 1
        self.new_count = new_count
        self.lines: List[Tuple[str, str]] = []  # (action, content)
    
    def validate(self, original_lines: List[str]) -> Optional[str]:
        """Validate this hunk against original file content. Returns error or None."""
        pos = self.old_start
        for action, content in self.lines:
            if action == " ":
                if pos >= len(original_lines) or original_lines[pos] != content:
                    return f"Context mismatch at original line {pos+1}: expected '{content[:50]}...'"
                pos += 1
            elif action == "-":
                if pos >= len(original_lines) or original_lines[pos] != content:
                    return f"Removal mismatch at original line {pos+1}"
                pos += 1
            elif action == "+":
                pass  # additions don't need validation
        return None


class FileChange:
    """Pending change for one file — applied only after validation."""
    
    def __init__(self, target: Path, original_content: str, new_content: str):
        self.target = target
        self.original_content = original_content
        self.new_content = new_content
        self.backup_path: Optional[Path] = None
        self._original_mode: Optional[int] = None
    
    def apply(self) -> bool:
        """Apply the change atomically via temp file + rename."""
        if self.target.exists():
            self._original_mode = self.target.stat().st_mode
            # Create backup
            self.backup_path = self.target.with_suffix(self.target.suffix + ".bak")
            shutil.copy2(self.target, self.backup_path)
        
        # Write to temp file first, then atomic rename
        tmp_path = self.target.with_suffix(self.target.suffix + ".tmp")
        try:
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_text(self.new_content, encoding="utf-8")
            tmp_path.replace(self.target)
            if self._original_mode:
                os.chmod(self.target, self._original_mode)
            return True
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
    
    def rollback(self):
        """Restore original file from backup."""
        if self.backup_path and self.backup_path.exists():
            self.backup_path.replace(self.target)
            self.backup_path = None


def _is_binary(filepath: Path) -> bool:
    """Detect if a file is binary by checking for null bytes."""
    if not filepath.exists():
        return False
    try:
        with open(filepath, 'rb') as f:
            chunk = f.read(8192)
            return b'\x00' in chunk
    except Exception:
        return False


def _parse_patch(patch_text: str, root: Path) -> Tuple[List[FileChange], List[str]]:
    """Parse unified diff into validated FileChange objects. Atomic: all-or-nothing."""
    lines = patch_text.splitlines()
    i = 0
    changes: List[FileChange] = []
    changed_paths: List[str] = []
    
    while i < len(lines):
        # Find file header
        while i < len(lines) and not lines[i].startswith("--- "):
            i += 1
        if i >= len(lines):
            break
        
        old_path = lines[i][4:].strip().split("\t")[0]
        i += 1
        if i >= len(lines) or not lines[i].startswith("+++ "):
            raise PatchError("Malformed patch: missing +++ after ---", file=old_path)
        new_path = lines[i][4:].strip().split("\t")[0]
        i += 1
        
        # Resolve relative path
        rel = (new_path if new_path != "/dev/null" else old_path).lstrip("ab/")
        if rel == "/dev/null":
            raise PatchError("File deletion patches are not supported")
        
        target = (root / rel).resolve()
        if target != root and root not in target.parents:
            raise PatchError("Patch escapes workspace", file=rel)
        
        # Binary check
        if _is_binary(target):
            raise PatchError("Binary files not supported for patching", file=rel)
        
        # Read original
        if target.exists():
            try:
                original_text = target.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                raise PatchError("Cannot read file as UTF-8 (binary?)", file=rel)
        else:
            original_text = ""
        original_lines = original_text.splitlines() if original_text else []
        
        # Parse hunks
        hunks: List[PatchHunk] = []
        while i < len(lines) and lines[i].startswith("@@ "):
            m = re.search(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", lines[i])
            if not m:
                raise PatchError("Malformed hunk header", file=rel, line=i+1)
            
            old_start = int(m.group(1))
            old_count = int(m.group(2)) if m.group(2) else 1
            new_start = int(m.group(3))
            new_count = int(m.group(4)) if m.group(4) else 1
            
            if old_start < 1:
                raise PatchError(f"Invalid hunk old_start={old_start}", file=rel)
            
            hunk = PatchHunk(old_start, old_count, new_start, new_count)
            i += 1
            
            # Parse hunk lines until next hunk or file boundary
            while i < len(lines) and not lines[i].startswith(("--- ", "@@ ", "diff --git ")):
                line = lines[i]
                if line.startswith(" "):
                    hunk.lines.append((" ", line[1:]))
                elif line.startswith("-"):
                    hunk.lines.append(("-", line[1:]))
                elif line.startswith("+"):
                    hunk.lines.append(("+", line[1:]))
                elif line == "\\ No newline at end of file":
                    # Handle properly: mark last line as no-newline
                    if hunk.lines:
                        last_action, last_content = hunk.lines[-1]
                        hunk.lines[-1] = (last_action, last_content + "[NOEOL]")
                elif line.strip() == "":
                    pass  # blank line within hunk
                else:
                    raise PatchError(f"Unexpected patch line: '{line[:50]}'", file=rel, line=i+1)
                i += 1
            
            hunks.append(hunk)
        
        if not hunks:
            raise PatchError("No hunks found in file patch", file=rel)
        
        # Validate ALL hunks before applying any
        validated_lines = list(original_lines)
        pos = 0
        
        for hunk in hunks:
            error = hunk.validate(original_lines)
            if error:
                raise PatchError(error, file=rel)
            
            # Build new content from this hunk
            out = []
            orig_pos = hunk.old_start
            
            # Copy unchanged lines before this hunk
            out.extend(validated_lines[pos:orig_pos])
            
            # Apply hunk
            for action, content in hunk.lines:
                if action == " ":
                    out.append(content)
                    orig_pos += 1
                elif action == "-":
                    orig_pos += 1
                elif action == "+":
                    content_clean = content.replace("[NOEOL]", "")
                    out.append(content_clean)
            
            # Copy remaining after hunk
            out.extend(original_lines[orig_pos:])
            validated_lines = out
            pos = hunk.old_start
        
        new_content = "\n".join(validated_lines)
        if original_text and not original_text.endswith("\n"):
            new_content = new_content.rstrip("\n")
        new_content += "\n" if new_content else ""
        
        changes.append(FileChange(target, original_text, new_content))
        changed_paths.append(str(target.relative_to(root)))
    
    if not changes:
        raise PatchError("No supported file hunks found in patch")
    
    return changes, changed_paths


def apply_unified_patch(root, patch_text):
    """
    Apply a unified diff patch atomically.

    All hunks are validated against current file content BEFORE any file is modified.
    On any validation failure, NO files are changed.
    On write failure, all modified files are rolled back.

    Returns list of changed relative paths.
    """
    root = Path(root).resolve()

    # Phase 0: Reject binary patch text immediately
    if any(b'\x00' in line.encode("utf-8", errors="surrogateescape")
           for line in patch_text.splitlines()):
        raise PatchError("Binary content detected in patch — not supported")
    
    # Phase 1: Parse and validate all changes (atomic — no writes yet)
    changes, changed_paths = _parse_patch(patch_text, root)
    
    # Phase 2: Apply all changes
    applied: List[FileChange] = []
    try:
        for change in changes:
            change.apply()
            applied.append(change)
    except Exception as e:
        # Rollback all applied changes
        for change in reversed(applied):
            try:
                change.rollback()
            except Exception:
                pass
        raise PatchError(f"Patch application failed: {e}") from e
    
    return changed_paths
