"""
Phase 8: Meta Learning — Cross-task pattern learning (Learning to Learn).

Teaches the AI to:
  ✅ Recognize patterns across different coding tasks
  ✅ Build reusable "skill templates" from successful strategies
  ✅ Improve few-shot performance by learning from task structures
  ✅ Identify which approach works best for which task type
  ✅ Self-modify its own prompts and strategies via meta-optimization
"""

from __future__ import annotations
import json
import time
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable


@dataclass
class SkillTemplate:
    """A learned pattern → strategy mapping."""
    id: str
    pattern_type: str       # "bug_pattern", "refactor_pattern", "design_pattern", ...
    trigger_keywords: list[str]
    description: str
    strategy: str           # what approach to apply
    success_rate: float     # 0-1, updated over time
    use_count: int = 0
    examples: list[dict] = field(default_factory=list)  # {context, code, result}
    timestamp: float = field(default_factory=time.time)


@dataclass
class TaskProfile:
    """Profile of a task type for strategy selection."""
    task_type: str
    avg_complexity: float = 0.0
    best_strategy: str = ""
    strategy_scores: dict[str, float] = field(default_factory=dict)  # strategy → avg score
    sample_count: int = 0


class MetaLearner:
    """
    Meta-learning system: finds patterns ACROSS tasks and learns which
    strategies work best for which situations.
    """

    def __init__(self, store_path: Optional[str] = None):
        self.store_path = Path(store_path) if store_path else Path("data/meta_knowledge.json")
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.skills: dict[str, SkillTemplate] = {}
        self.task_profiles: dict[str, TaskProfile] = {}
        self._load()

    def register_skill(
        self,
        pattern_type: str,
        trigger_keywords: list[str],
        strategy: str,
        success: bool,
        context: str = "",
        code: str = "",
    ):
        """Register a new skill or update existing one."""
        key = hashlib.md5(f"{pattern_type}{' '.join(sorted(trigger_keywords))}".encode()).hexdigest()[:12]

        if key in self.skills:
            skill = self.skills[key]
            skill.use_count += 1
            # Update success rate with exponential moving average
            alpha = 0.1
            skill.success_rate = (1 - alpha) * skill.success_rate + alpha * (1.0 if success else 0.0)
            if context and code:
                skill.examples.append({"context": context, "code": code, "result": "success" if success else "failure"})
                if len(skill.examples) > 50:
                    skill.examples = skill.examples[-50:]
        else:
            skill = SkillTemplate(
                id=key,
                pattern_type=pattern_type,
                trigger_keywords=trigger_keywords,
                description=f"Pattern: {' '.join(trigger_keywords[:5])}",
                strategy=strategy,
                success_rate=1.0 if success else 0.0,
                examples=[{"context": context, "code": code, "result": "success" if success else "failure"}] if context else [],
            )
            self.skills[key] = skill

        self._save()

    def match_skills(
        self,
        problem_description: str,
        min_success_rate: float = 0.5,
        top_k: int = 5,
    ) -> list[SkillTemplate]:
        """Find relevant skills for a given problem."""
        desc_lower = problem_description.lower()
        matches = []

        for skill in self.skills.values():
            if skill.success_rate < min_success_rate:
                continue
            # Count keyword matches
            hits = sum(1 for kw in skill.trigger_keywords if kw.lower() in desc_lower)
            if hits > 0:
                score = hits * skill.success_rate
                matches.append((score, skill))

        matches.sort(key=lambda x: -x[0])
        return [m for _, m in matches[:top_k]]

    def suggest_strategy(self, task_type: str, problem_description: str = "") -> str:
        """Suggest the best strategy for a given task."""
        # First, check task profiles
        profile = self.task_profiles.get(task_type)
        if profile and profile.best_strategy and profile.sample_count >= 10:
            return profile.best_strategy

        # Fall back to skill matching
        skills = self.match_skills(problem_description, min_success_rate=0.3, top_k=1)
        if skills:
            return skills[0].strategy

        return "general_approach"

    def record_task_result(
        self,
        task_type: str,
        strategy: str,
        score: float,
        complexity: float = 5.0,
    ):
        """Update task profile with result of a strategy."""
        if task_type not in self.task_profiles:
            self.task_profiles[task_type] = TaskProfile(task_type=task_type)

        profile = self.task_profiles[task_type]
        profile.sample_count += 1

        # Update strategy scores
        if strategy not in profile.strategy_scores:
            profile.strategy_scores[strategy] = score
        else:
            alpha = 0.2
            profile.strategy_scores[strategy] = (
                (1 - alpha) * profile.strategy_scores[strategy] + alpha * score
            )

        # Update best strategy
        profile.best_strategy = max(
            profile.strategy_scores,
            key=lambda s: profile.strategy_scores[s],
        )

        # Update avg complexity
        alpha = 0.1
        profile.avg_complexity = (1 - alpha) * profile.avg_complexity + alpha * complexity

        self._save()

    def get_strategy_ranking(self, task_type: str) -> list[tuple[str, float]]:
        """Get strategies ranked by effectiveness for a task type."""
        profile = self.task_profiles.get(task_type)
        if not profile:
            return []
        return sorted(profile.strategy_scores.items(), key=lambda x: -x[1])

    def few_shot_prompt(
        self,
        task_type: str,
        task_description: str,
        n_examples: int = 3,
    ) -> str:
        """Generate a few-shot prompt using past successful examples."""
        skills = self.match_skills(task_description, min_success_rate=0.6, top_k=n_examples)

        if not skills:
            return task_description

        prompt_parts = ["# Task", task_description, "", "# Examples of successful approaches:"]
        for i, skill in enumerate(skills, 1):
            prompt_parts.append(f"\n## Example {i}")
            prompt_parts.append(f"Strategy: {skill.strategy[:300]}")
            if skill.examples:
                ex = skill.examples[0]
                prompt_parts.append(f"Context: {ex.get('context', '')[:200]}")
                prompt_parts.append(f"```\n{ex.get('code', '')[:500]}\n```")
            prompt_parts.append(f"Success rate: {skill.success_rate:.0%}")

        return "\n".join(prompt_parts)

    def meta_optimize_prompt(
        self,
        task_type: str,
        current_prompt: str,
        evaluate_fn: Callable[[str], float],
        iterations: int = 5,
    ) -> tuple[str, float]:
        """
        Meta-optimize a prompt template by trying variations.

        Args:
            task_type: Type of task this prompt is for
            current_prompt: Current prompt template
            evaluate_fn: Function that evaluates a prompt → score
            iterations: Number of optimization iterations

        Returns:
            (best_prompt, best_score)
        """
        best_prompt = current_prompt
        best_score = evaluate_fn(current_prompt)
        strategies = self.get_strategy_ranking(task_type)

        for _ in range(iterations):
            # Try incorporating one of the top strategies into the prompt
            for strategy_name, strategy_score in strategies[:3]:
                if strategy_name.lower() not in best_prompt.lower():
                    candidate = best_prompt + f"\n\nTip: Consider using the '{strategy_name}' approach."
                    score = evaluate_fn(candidate)
                    if score > best_score:
                        best_score = score
                        best_prompt = candidate
                        break  # Take first improvement, re-evaluate next iter
            else:
                break  # No improvement found

        return best_prompt, best_score

    def _save(self):
        data = {
            "skills": {
                k: {
                    "id": s.id, "pattern_type": s.pattern_type,
                    "trigger_keywords": s.trigger_keywords,
                    "description": s.description, "strategy": s.strategy,
                    "success_rate": s.success_rate, "use_count": s.use_count,
                    "examples": s.examples[-20:], "timestamp": s.timestamp,
                }
                for k, s in self.skills.items()
            },
            "task_profiles": {
                k: {
                    "task_type": p.task_type,
                    "avg_complexity": p.avg_complexity,
                    "best_strategy": p.best_strategy,
                    "strategy_scores": p.strategy_scores,
                    "sample_count": p.sample_count,
                }
                for k, p in self.task_profiles.items()
            },
        }
        self.store_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def _load(self):
        if not self.store_path.exists():
            return
        data = json.loads(self.store_path.read_text())
        for k, s in data.get("skills", {}).items():
            self.skills[k] = SkillTemplate(
                id=s["id"], pattern_type=s["pattern_type"],
                trigger_keywords=s["trigger_keywords"],
                description=s["description"], strategy=s["strategy"],
                success_rate=s["success_rate"], use_count=s.get("use_count", 0),
                examples=s.get("examples", []), timestamp=s.get("timestamp", 0),
            )
        for k, p in data.get("task_profiles", {}).items():
            self.task_profiles[k] = TaskProfile(
                task_type=p["task_type"],
                avg_complexity=p.get("avg_complexity", 0.0),
                best_strategy=p.get("best_strategy", ""),
                strategy_scores=p.get("strategy_scores", {}),
                sample_count=p.get("sample_count", 0),
            )
