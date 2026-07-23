"""
Phase 8: Memory Evolution — Experience Replay, Failure DB, Knowledge Compression.

Enables the AI to:
  ✅ Remember past successes and replicate them
  ✅ Learn from failures via a searchable error database
  ✅ Compress and summarize long conversation/action histories
  ✅ Build a growing repository knowledge graph
  ✅ Prioritize important memories (importance-weighted recall)
"""

from __future__ import annotations
import json
import time
import hashlib
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Experience:
    """A single experience (success or failure)."""
    id: str
    task_type: str          # "code_generation", "bug_fix", "refactor", "design", ...
    context: str            # brief description of what was attempted
    outcome: str            # "success", "failure", "partial"
    score: float            # 0-100 evaluation score
    strategy: str           # what approach was used
    error_pattern: str = "" # extracted error pattern (for failures)
    fix_pattern: str = ""   # what fixed it (for successes after failure)
    code_snippet: str = ""  # relevant code
    tags: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    replay_count: int = 0


@dataclass
class FailurePattern:
    """Recognized pattern of failure — used for proactive avoidance."""
    pattern_hash: str
    description: str
    occurrence_count: int = 0
    last_seen: float = 0.0
    common_fixes: list[str] = field(default_factory=list)
    affected_files: list[str] = field(default_factory=list)


class FailureDatabase:
    """Searchable database of past failures for rapid diagnosis."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else Path("data/failure_db.jsonl")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.patterns: dict[str, FailurePattern] = {}
        self._load()

    def record_failure(
        self,
        error_message: str,
        file: str = "",
        line: int = 0,
        fix: str = "",
    ):
        """Record a failure for future reference."""
        pattern_hash = hashlib.md5(error_message.encode()).hexdigest()[:12]

        if pattern_hash in self.patterns:
            pat = self.patterns[pattern_hash]
            pat.occurrence_count += 1
            pat.last_seen = time.time()
            if fix and fix not in pat.common_fixes:
                pat.common_fixes.append(fix)
            if file and file not in pat.affected_files:
                pat.affected_files.append(file)
        else:
            self.patterns[pattern_hash] = FailurePattern(
                pattern_hash=pattern_hash,
                description=error_message[:500],
                occurrence_count=1,
                last_seen=time.time(),
                common_fixes=[fix] if fix else [],
                affected_files=[file] if file else [],
            )

        self._save()

    def lookup(self, error_message: str, top_k: int = 5) -> list[FailurePattern]:
        """Find similar past failures."""
        query_hash = hashlib.md5(error_message.encode()).hexdigest()[:12]

        # Exact match
        if query_hash in self.patterns:
            return [self.patterns[query_hash]]

        # Fuzzy: substring match
        results = []
        for pat in self.patterns.values():
            # Simple word overlap
            query_words = set(error_message.lower().split())
            pat_words = set(pat.description.lower().split())
            overlap = len(query_words & pat_words)
            if overlap > 0:
                results.append((overlap, pat))

        results.sort(key=lambda x: -x[0])
        return [p for _, p in results[:top_k]]

    def get_common_fixes(self, error_message: str) -> list[str]:
        """Get known fixes for a given error pattern."""
        matches = self.lookup(error_message)
        fixes = []
        for m in matches:
            fixes.extend(m.common_fixes)
        return list(dict.fromkeys(fixes))  # unique, preserve order

    def _save(self):
        with open(self.db_path, "w") as f:
            for pat in self.patterns.values():
                f.write(json.dumps({
                    "hash": pat.pattern_hash,
                    "description": pat.description,
                    "count": pat.occurrence_count,
                    "last_seen": pat.last_seen,
                    "fixes": pat.common_fixes,
                    "files": pat.affected_files,
                }, ensure_ascii=False) + "\n")

    def _load(self):
        if not self.db_path.exists():
            return
        with open(self.db_path) as f:
            for line in f:
                d = json.loads(line)
                self.patterns[d["hash"]] = FailurePattern(
                    pattern_hash=d["hash"],
                    description=d["description"],
                    occurrence_count=d["count"],
                    last_seen=d["last_seen"],
                    common_fixes=d["fixes"],
                    affected_files=d["files"],
                )


class ExperienceReplay:
    """Stores and replays experiences for continual learning."""

    def __init__(self, capacity: int = 10000, store_path: Optional[str] = None):
        self.capacity = capacity
        self.store_path = Path(store_path) if store_path else Path("data/experience_replay.jsonl")
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.experiences: list[Experience] = []
        self._by_task: dict[str, list[int]] = defaultdict(list)
        self._load()

    def add(self, exp: Experience):
        """Add an experience, evicting old ones if needed."""
        exp.id = hashlib.md5(f"{exp.task_type}{exp.context}{time.time()}".encode()).hexdigest()[:16]

        if len(self.experiences) >= self.capacity:
            # Evict least replayed, lowest score
            self.experiences.sort(key=lambda e: (e.replay_count, e.score))
            self.experiences.pop(0)

        self.experiences.append(exp)
        self._by_task[exp.task_type].append(len(self.experiences) - 1)
        self._save()

    def sample(
        self,
        task_type: Optional[str] = None,
        outcome: Optional[str] = None,
        n: int = 5,
        prioritize_recent: bool = True,
        prioritize_high_score: bool = False,
    ) -> list[Experience]:
        """Sample experiences for replay, with optional filtering."""
        pool = self.experiences

        if task_type:
            indices = self._by_task.get(task_type, [])
            pool = [self.experiences[i] for i in indices if i < len(self.experiences)]

        if outcome:
            pool = [e for e in pool if e.outcome == outcome]

        if not pool:
            return []

        # Scoring for prioritization
        def priority(e: Experience) -> float:
            s = 0.0
            if prioritize_recent:
                s += 1.0 / (1.0 + (time.time() - e.timestamp) / 86400)  # recency
            if prioritize_high_score:
                s += e.score / 100.0
            s += e.replay_count * 0.01  # slight boost for frequently replayed
            return s

        pool = sorted(pool, key=priority, reverse=True)

        # Mark as replayed
        for e in pool[:n]:
            e.replay_count += 1

        return pool[:n]

    def get_successful_strategies(self, task_type: str) -> list[str]:
        """Get strategies that worked for a given task type."""
        successes = self.sample(task_type=task_type, outcome="success", n=20)
        strategies = list(dict.fromkeys(e.strategy for e in successes if e.strategy))
        return strategies[:10]

    def get_failure_lessons(self, task_type: str) -> list[str]:
        """Get lessons from past failures."""
        failures = self.sample(task_type=task_type, outcome="failure", n=20)
        return [f"❌ {e.context}: {e.error_pattern}" for e in failures if e.error_pattern][:10]

    def _save(self):
        with open(self.store_path, "w") as f:
            for e in self.experiences[-1000:]:  # save last 1000 to disk
                f.write(json.dumps({
                    "id": e.id, "task_type": e.task_type,
                    "context": e.context, "outcome": e.outcome,
                    "score": e.score, "strategy": e.strategy,
                    "error_pattern": e.error_pattern,
                    "fix_pattern": e.fix_pattern,
                    "tags": e.tags, "timestamp": e.timestamp,
                }, ensure_ascii=False) + "\n")

    def _load(self):
        if not self.store_path.exists():
            return
        with open(self.store_path) as f:
            for line in f:
                d = json.loads(line)
                exp = Experience(
                    id=d.get("id", ""),
                    task_type=d["task_type"],
                    context=d["context"],
                    outcome=d["outcome"],
                    score=d["score"],
                    strategy=d["strategy"],
                    error_pattern=d.get("error_pattern", ""),
                    fix_pattern=d.get("fix_pattern", ""),
                    tags=d.get("tags", []),
                    timestamp=d["timestamp"],
                )
                self.experiences.append(exp)
                self._by_task[exp.task_type].append(len(self.experiences) - 1)


class KnowledgeCompressor:
    """Compresses long histories into compact summaries for context windows."""

    def __init__(self, max_tokens: int = 2048):
        self.max_tokens = max_tokens

    def compress_conversation(self, messages: list[dict]) -> str:
        """Compress a conversation into a dense summary."""
        if not messages:
            return ""

        # Extract key points
        key_points = []
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "unknown")

            # Simple heuristic: keep action items, decisions, errors
            if any(kw in content.lower() for kw in
                   ["error", "fix", "decided", "implement", "test", "fail", "pass",
                    "change", "create", "delete", "modify", "important"]):
                short = content[:200]
                key_points.append(f"[{role}] {short}")

        if len(key_points) == 0 and messages:
            # Fallback: first and last message
            key_points = [
                f"[{messages[0].get('role','')}] {messages[0].get('content','')[:150]}",
            ]
            if len(messages) > 1:
                key_points.append(
                    f"[{messages[-1].get('role','')}] {messages[-1].get('content','')[:150]}",
                )

        return "\n".join(key_points)

    def compress_code_changes(self, changes: list[dict]) -> str:
        """Summarize a set of code changes."""
        lines = []
        for change in changes:
            fname = change.get("file", "unknown")
            action = change.get("action", "modified")
            summary = change.get("summary", "")
            lines.append(f"- {action} {fname}: {summary[:100]}")
        return "\n".join(lines)

    def extract_knowledge(self, experiences: list[Experience]) -> str:
        """Extract compressed knowledge from experiences."""
        if not experiences:
            return "No prior knowledge."

        successes = [e for e in experiences if e.outcome == "success"]
        failures = [e for e in experiences if e.outcome == "failure"]

        lines = ["## Prior Knowledge"]

        if successes:
            lines.append(f"\n### Successful strategies ({len(successes)}):")
            for e in successes[:5]:
                lines.append(f"- ✓ [{e.task_type}] {e.strategy[:200]} (score: {e.score})")

        if failures:
            lines.append(f"\n### Lessons from failures ({len(failures)}):")
            for e in failures[:5]:
                lines.append(f"- ✗ [{e.task_type}] {e.error_pattern[:200]}")

        return "\n".join(lines)
