"""Critic/Evaluator/Fixer orchestration for TinyLLM Coding Agent."""
from __future__ import annotations
import json

class ReviewPipeline:
    def __init__(self, provider, tools, graph=None):
        self.provider=provider; self.tools=tools; self.graph=graph

    def _ask(self, system, prompt):
        return self.provider.generate(prompt, system).text.strip()

    def critic(self, task, diff):
        return self._ask(
          "You are a strict senior code reviewer. Find correctness, security, regression and test gaps. "
          "Return JSON: {\"issues\":[{\"severity\":\"high|medium|low\",\"file\":\"...\",\"reason\":\"...\"}],\"summary\":\"...\"}.",
          f"Task:\n{task}\n\nGit diff:\n{diff}")

    def evaluate(self, task, diff, tests):
        return self._ask(
          "You are an independent release evaluator. Judge whether the task is actually complete. "
          "Return JSON: {\"score\":0-100,\"pass\":true|false,\"missing\":[...],\"reason\":\"...\"}.",
          f"Task:\n{task}\n\nDiff:\n{diff}\n\nVerification:\n{tests}")

    def review(self, task):
        diff=self.tools.execute("git_diff",{}).get("diff","")
        tests=self.tools.execute("run_tests",{})
        critic=self.critic(task,diff)
        evaluation=self.evaluate(task,diff,tests)
        return {"critic":critic,"tests":tests,"evaluation":evaluation}
