"""
benchmark.py — Evaluation framework for TinyLLM code generation & repair.

Features:
  - Task suite: define coding tasks with test cases
  - Runner: execute model-generated code in sandbox
  - Scorer: pass/fail + execution metrics
  - Reporter: JSON + Markdown summaries
  - Regression: compare runs over time
"""
import json, time, os, sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field, asdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sandbox import CodeSandbox, SandboxResult


@dataclass
class TestCase:
    """A single test case within a task."""
    name: str
    input: str = ""
    expected_output: str = ""
    expected_exit_code: int = 0
    timeout_sec: int = 10

@dataclass
class BenchmarkTask:
    """A coding task for evaluation."""
    id: str
    name: str
    description: str
    language: str = "python"
    test_cases: List[TestCase] = field(default_factory=list)
    starter_code: str = ""
    expected_patterns: List[str] = field(default_factory=list)  # must-appear patterns

@dataclass
class TaskResult:
    """Result of running a single task."""
    task_id: str
    task_name: str
    success: bool = False
    passed_tests: int = 0
    total_tests: int = 0
    runtime_ms: float = 0
    test_results: List[dict] = field(default_factory=list)
    error: str = ""


class BenchmarkRunner:
    """Execute benchmark tasks and score results."""

    def __init__(self, sandbox_timeout: int = 30):
        self.sandbox_timeout = sandbox_timeout
        self.results: List[TaskResult] = []

    # ── Run ───────────────────────────────────────────────

    def run_task(self, task: BenchmarkTask, generated_code: str) -> TaskResult:
        """Run a single task with the given generated code."""
        result = TaskResult(task_id=task.id, task_name=task.name,
                            total_tests=len(task.test_cases))
        t0 = time.time()

        sandbox = CodeSandbox(timeout=self.sandbox_timeout)
        try:
            # Write the generated code
            sandbox.write_file("solution.py", generated_code)

            # Optional: pattern check
            for pattern in task.expected_patterns:
                if pattern not in generated_code:
                    result.error = f"Missing expected pattern: {pattern}"
                    result.runtime_ms = (time.time() - t0) * 1000
                    return result

            # Run each test case
            for tc in task.test_cases:
                r = sandbox.run_python_file("solution.py", stdin=tc.input)
                passed = (r.exit_code == tc.expected_exit_code and
                          tc.expected_output in r.stdout)

                result.test_results.append({
                    "name": tc.name,
                    "passed": passed,
                    "exit_code": r.exit_code,
                    "expected_exit": tc.expected_exit_code,
                    "stdout": r.stdout[:300],
                    "expected": tc.expected_output[:300],
                    "runtime_ms": r.runtime_ms,
                })
                if passed:
                    result.passed_tests += 1

            result.success = (result.passed_tests == result.total_tests)

        except Exception as e:
            result.error = str(e)
        finally:
            sandbox.cleanup()

        result.runtime_ms = (time.time() - t0) * 1000
        self.results.append(result)
        return result

    def run_suite(self, tasks: List[BenchmarkTask],
                  generate_fn: Callable[[BenchmarkTask], str]) -> List[TaskResult]:
        """Run a full benchmark suite."""
        self.results = []
        for task in tasks:
            print(f"  📋 {task.id}: {task.name} ...", end=" ")
            code = generate_fn(task)
            result = self.run_task(task, code)
            status = "✅" if result.success else "❌"
            print(f"{status} ({result.passed_tests}/{result.total_tests} tests)")
        return self.results

    # ── Scoring ───────────────────────────────────────────

    def score(self) -> dict:
        """Compute aggregate scores."""
        if not self.results:
            return {"pass_rate": 0, "total": 0, "passed": 0}

        total = len(self.results)
        passed = sum(1 for r in self.results if r.success)
        total_tests = sum(r.total_tests for r in self.results)
        passed_tests = sum(r.passed_tests for r in self.results)
        avg_time = sum(r.runtime_ms for r in self.results) / total if total else 0

        return {
            "tasks_total": total,
            "tasks_passed": passed,
            "task_pass_rate": round(passed / total * 100, 1) if total else 0,
            "tests_total": total_tests,
            "tests_passed": passed_tests,
            "test_pass_rate": round(passed_tests / total_tests * 100, 1) if total_tests else 0,
            "avg_runtime_ms": round(avg_time, 1),
        }

    # ── Report ────────────────────────────────────────────

    def report_json(self) -> str:
        return json.dumps({
            "score": self.score(),
            "results": [asdict(r) for r in self.results],
        }, indent=2, ensure_ascii=False)

    def report_markdown(self) -> str:
        score = self.score()
        lines = [
            "# Benchmark Report",
            f"**Pass Rate**: {score['task_pass_rate']}% ({score['tasks_passed']}/{score['tasks_total']} tasks)",
            f"**Test Pass Rate**: {score['test_pass_rate']}% ({score['tests_passed']}/{score['tests_total']} tests)",
            f"**Avg Runtime**: {score['avg_runtime_ms']:.0f}ms",
            "",
            "| Task | Status | Tests | Time |",
            "|------|--------|-------|------|",
        ]
        for r in self.results:
            status = "✅" if r.success else "❌"
            lines.append(
                f"| {r.task_name} | {status} | {r.passed_tests}/{r.total_tests} | {r.runtime_ms:.0f}ms |"
            )
        return '\n'.join(lines)

    def compare_runs(self, previous_report_path: str) -> dict:
        """Compare current run against a previous report."""
        try:
            with open(previous_report_path) as f:
                prev = json.load(f)
            curr = self.score()
            prev_score = prev.get("score", {})

            return {
                "current": curr,
                "previous": prev_score,
                "delta_task_pass": round(curr["task_pass_rate"] - prev_score.get("task_pass_rate", 0), 1),
                "delta_test_pass": round(curr["test_pass_rate"] - prev_score.get("test_pass_rate", 0), 1),
                "regression": curr["task_pass_rate"] < prev_score.get("task_pass_rate", 0),
            }
        except Exception as e:
            return {"error": str(e)}


# ═══════════════════════════════════════════════════════════
# Built-in Benchmark Tasks
# ═══════════════════════════════════════════════════════════

BUILTIN_TASKS = [
    BenchmarkTask(
        id="fibonacci",
        name="Fibonacci Function",
        description="Write a function fib(n) that returns the nth Fibonacci number (0-indexed).",
        test_cases=[
            TestCase("fib0", expected_output="0"),
            TestCase("fib1", expected_output="1"),
            TestCase("fib5", expected_output="5"),
            TestCase("fib10", expected_output="55"),
            TestCase("fib20", expected_output="6765"),
        ],
        starter_code="""# Write fib(n) here
def fib(n):
    pass

# Test runner
import sys
for line in sys.stdin:
    n = int(line.strip())
    print(fib(n))
""",
        expected_patterns=["def fib", "return"],
    ),
    BenchmarkTask(
        id="fizzbuzz",
        name="FizzBuzz",
        description="Print numbers 1 to N. Multiples of 3 → 'Fizz', 5 → 'Buzz', both → 'FizzBuzz'.",
        test_cases=[
            TestCase("n3", input="3\n", expected_output="1\n2\nFizz"),
            TestCase("n5", input="5\n", expected_output="1\n2\nFizz\n4\nBuzz"),
            TestCase("n15", input="15\n",
                     expected_output="1\n2\nFizz\n4\nBuzz\nFizz\n7\n8\nFizz\nBuzz\n11\nFizz\n13\n14\nFizzBuzz"),
        ],
        starter_code="""import sys
n = int(sys.stdin.readline().strip())
# Your code here
""",
        expected_patterns=["Fizz", "Buzz", "%", "for"],
    ),
    BenchmarkTask(
        id="reverse_string",
        name="String Reverse",
        description="Write a function that reverses a string.",
        test_cases=[
            TestCase("hello", expected_output="olleh"),
            TestCase("empty", input="\n", expected_output=""),
            TestCase("unicode", input="日本語\n", expected_output="語本日"),
        ],
        starter_code="""import sys
s = sys.stdin.readline().rstrip('\\n')
# Print reversed string
""",
        expected_patterns=["[::-1]", "reversed"],
    ),
    BenchmarkTask(
        id="binary_search",
        name="Binary Search",
        description="Implement binary search. Return index or -1 if not found.",
        test_cases=[
            TestCase("found", input="1 2 3 4 5\n3\n", expected_output="2"),
            TestCase("notfound", input="1 2 3 4 5\n6\n", expected_output="-1"),
            TestCase("empty", input="\n1\n", expected_output="-1"),
        ],
        starter_code="""import sys
arr = list(map(int, sys.stdin.readline().split()))
target = int(sys.stdin.readline())
# Your binary search here
""",
        expected_patterns=["while", "mid", "//"],
    ),
    BenchmarkTask(
        id="error_handling",
        name="Error Handling",
        description="Write safe division: return result or error message.",
        test_cases=[
            TestCase("normal", input="10 2\n", expected_output="5"),
            TestCase("zero", input="10 0\n", expected_output="Error"),
            TestCase("invalid", input="abc 2\n", expected_output="Error"),
        ],
        starter_code="""import sys
line = sys.stdin.readline().strip()
# Parse a and b, handle errors
""",
        expected_patterns=["try", "except", "ZeroDivisionError", "ValueError"],
    ),
]


# ── CLI ───────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: benchmark.py <run|report> [--output file.json]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "run":
        # Simple demo: use a template-based generator
        def demo_generate(task: BenchmarkTask) -> str:
            return task.starter_code  # Would be replaced by LLM

        runner = BenchmarkRunner()
        runner.run_suite(BUILTIN_TASKS, demo_generate)
        score = runner.score()
        print(f"\nScore: {score['task_pass_rate']}% ({score['tasks_passed']}/{score['tasks_total']})")
        print(runner.report_markdown())

    elif cmd == "report":
        runner = BenchmarkRunner()
        # Load previous results if available
        print(runner.report_json())
