
import re
from pathlib import Path

class PatchError(Exception):
    pass

def apply_unified_patch(root, patch_text):
    """
    Minimal safe unified-diff applier for file edits.
    Supports standard ---/+++ and @@ hunks.
    Returns changed relative paths.
    """
    root=Path(root).resolve()
    lines=patch_text.splitlines()
    i=0
    changed=[]
    while i < len(lines):
        if not lines[i].startswith("--- "):
            i+=1; continue
        old=lines[i][4:].strip().split("\t")[0]
        i+=1
        if i>=len(lines) or not lines[i].startswith("+++ "):
            raise PatchError("Malformed patch: missing +++")
        new=lines[i][4:].strip().split("\t")[0]
        i+=1
        rel=(new if new!=" /dev/null" else old).lstrip("ab/")
        if rel=="/dev/null": raise PatchError("Deletion-only patch is not supported")
        target=(root/rel).resolve()
        if target!=root and root not in target.parents: raise PatchError("Patch escapes workspace")
        original=target.read_text(encoding="utf-8").splitlines() if target.exists() else []
        out=[]
        pos=0
        while i<len(lines) and lines[i].startswith("@@ "):
            m=re.search(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", lines[i])
            if not m: raise PatchError("Malformed hunk header")
            old_start=int(m.group(1))-1
            i+=1
            if old_start<pos or old_start>len(original): raise PatchError("Hunk location out of range")
            out.extend(original[pos:old_start]); pos=old_start
            while i<len(lines) and not lines[i].startswith(("--- ","@@ ")):
                line=lines[i]
                if line.startswith(" "):
                    content=line[1:]
                    if pos>=len(original) or original[pos]!=content: raise PatchError("Context mismatch")
                    out.append(content); pos+=1
                elif line.startswith("-"):
                    content=line[1:]
                    if pos>=len(original) or original[pos]!=content: raise PatchError("Removal mismatch")
                    pos+=1
                elif line.startswith("+"):
                    out.append(line[1:])
                elif line=="\\ No newline at end of file":
                    pass
                else:
                    raise PatchError("Unsupported patch line")
                i+=1
        out.extend(original[pos:])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(out)+("\n" if out else ""), encoding="utf-8")
        changed.append(str(target.relative_to(root)))
    if not changed: raise PatchError("No supported file hunks found")
    return changed
