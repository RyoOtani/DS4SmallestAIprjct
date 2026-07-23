"""
Phase 6: Critic & Debugger Agent — Automated code review and debugging.
"""

from __future__ import annotations
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .code_understanding import RepositoryAnalyzer
from .quality import QualityIssue, Linter, StaticAnalyzer


@dataclass
class ReviewComment:
    file: str
    line: int
    severity: str  # blocker, major, minor, nit, praise
    category: str  # performance, security, readability, correctness, style, architecture
    message: str
    suggestion: str = ""


@dataclass
class CodeReview:
    comments: list[ReviewComment] = field(default_factory=list)
    score: float = 5.0  # 0-10
    summary: str = ""
    
    @property
    def blockers(self) -> list[ReviewComment]:
        return [c for c in self.comments if c.severity == "blocker"]
    
    @property
    def passed(self) -> bool:
        return len(self.blockers) == 0


class CriticAgent:
    """Automated code reviewer — checks correctness, style, security, performance."""
    
    def __init__(self, analyzer: Optional[RepositoryAnalyzer] = None):
        self.analyzer = analyzer
        self.linter = Linter()
        self.static = StaticAnalyzer()
    
    def review_file(self, filepath: str) -> CodeReview:
        """Perform a comprehensive code review of a file."""
        review = CodeReview()
        
        try:
            source = Path(filepath).read_text(encoding="utf-8")
        except Exception:
            review.comments.append(ReviewComment(
                file=filepath, line=0, severity="blocker",
                category="correctness", message="Cannot read file.",
            ))
            review.score = 0.0
            return review
        
        lang = self._detect_lang(filepath)
        lines = source.splitlines()
        
        # 1. Linting
        lint_report = self.linter.lint_file(filepath)
        for issue in lint_report.issues:
            sev_map = {"error": "blocker", "warning": "minor"}
            review.comments.append(ReviewComment(
                file=filepath, line=issue.line,
                severity=sev_map.get(issue.severity, "minor"),
                category="correctness" if issue.severity == "error" else "style",
                message=issue.message,
            ))
        
        # 2. Static analysis
        static_report = self.static.analyze_file(filepath)
        for issue in static_report.issues:
            cat_map = {
                "hardcoded-secret": "security",
                "private-key": "security",
                "dangerous-eval": "security",
                "dangerous-exec": "security",
                "dangerous-system-call": "security",
            }
            rev = ReviewComment(
                file=filepath, line=issue.line,
                severity="blocker" if "secret" in issue.rule or "dangerous" in issue.rule else "major",
                category=cat_map.get(issue.rule, "correctness"),
                message=issue.message,
                suggestion=issue.suggestion,
            )
            review.comments.append(rev)
        
        # 3. Best practice checks
        review = self._check_best_practices(source, lines, filepath, lang, review)
        
        # 4. Score calculation
        n_blockers = len(review.blockers)
        n_major = sum(1 for c in review.comments if c.severity == "major")
        n_minor = sum(1 for c in review.comments if c.severity == "minor")
        review.score = max(0.0, 10.0 - n_blockers * 3.0 - n_major * 1.0 - n_minor * 0.3)
        review.score = min(10.0, round(review.score, 1))
        
        return review
    
    def review_directory(self, path: str) -> dict[str, CodeReview]:
        """Review all files in a directory."""
        results = {}
        root = Path(path)
        skip_dirs = {'.git', '__pycache__', 'node_modules', 'build', 'dist', '.venv'}
        extensions = {'.py', '.c', '.h', '.cpp', '.js', '.ts', '.jsx', '.tsx'}
        
        for f in root.rglob('*'):
            if any(s in f.parts for s in skip_dirs):
                continue
            if f.suffix in extensions:
                results[str(f)] = self.review_file(str(f))
        
        return results
    
    def _check_best_practices(self, source: str, lines: list[str], filepath: str, lang: str, review: CodeReview) -> CodeReview:
        """Language-agnostic best practice checks."""
        
        # Function length
        if lang == 'python':
            import ast as py_ast
            try:
                tree = py_ast.parse(source)
                for node in py_ast.walk(tree):
                    if isinstance(node, (py_ast.FunctionDef, py_ast.AsyncFunctionDef)):
                        func_lines = node.end_lineno - node.lineno if node.end_lineno else 0
                        if func_lines > 80:
                            review.comments.append(ReviewComment(
                                file=filepath, line=node.lineno, severity="major",
                                category="readability",
                                message=f"Function '{node.name}' is {func_lines} lines — too long.",
                                suggestion="Split into smaller functions (< 50 lines each).",
                            ))
                        if not py_ast.get_docstring(node) and func_lines > 20:
                            review.comments.append(ReviewComment(
                                file=filepath, line=node.lineno, severity="minor",
                                category="readability",
                                message=f"Function '{node.name}' has no docstring.",
                                suggestion="Add a descriptive docstring.",
                            ))
            except SyntaxError:
                pass
        
        # Line length
        long_lines = [(i, len(l)) for i, l in enumerate(lines, 1) if len(l) > 120]
        if len(long_lines) > 5:
            review.comments.append(ReviewComment(
                file=filepath, line=long_lines[0][0], severity="minor",
                category="style",
                message=f"{len(long_lines)} lines exceed 120 characters.",
                suggestion="Break long lines or extract expressions.",
            ))
        
        # File-level
        if len(lines) > 600:
            review.comments.append(ReviewComment(
                file=filepath, line=1, severity="major",
                category="architecture",
                message=f"File is {len(lines)} lines — too large.",
                suggestion="Split into multiple modules by responsibility.",
            ))
        
        return review
    
    @staticmethod
    def _detect_lang(filepath: str) -> str:
        ext = Path(filepath).suffix.lower()
        lang_map = {
            '.py': 'python', '.c': 'c', '.h': 'c',
            '.cpp': 'cpp', '.js': 'javascript', '.ts': 'typescript',
            '.jsx': 'javascript', '.tsx': 'typescript',
        }
        return lang_map.get(ext, 'unknown')


class DebuggerAgent:
    """Analyzes test failures, build errors, and proposes fixes."""
    
    STATUS_OK = "ok"
    STATUS_BUILD_FAIL = "build_fail"
    STATUS_TEST_FAIL = "test_fail"
    STATUS_RUNTIME_ERROR = "runtime_error"
    
    @dataclass
    class DebugResult:
        status: str
        diagnosis: str = ""
        error_file: str = ""
        error_line: int = 0
        error_type: str = ""
        stack_trace: str = ""
        suggestions: list[str] = field(default_factory=list)
    
    def analyze_build_error(self, build_output: str) -> DebugResult:
        """Analyze build/compile errors."""
        result = self.DebugResult(status=self.STATUS_BUILD_FAIL)
        
        # GCC/Clang error format: file:line:col: error: message
        for m in re.finditer(r'(.+):(\d+):(\d+):\s+(error|warning):\s+(.+)', build_output):
            result.error_file = m.group(1)
            result.error_line = int(m.group(2))
            result.error_type = "compile_error" if m.group(4) == "error" else "compile_warning"
            result.diagnosis = m.group(5)
            
            # Suggest fixes
            msg = m.group(5).lower()
            if "undefined reference" in msg or "implicit declaration" in msg:
                result.suggestions.append(f"Missing include or function declaration in '{result.error_file}'")
            elif "expected" in msg and "before" in msg:
                result.suggestions.append(f"Syntax error at line {result.error_line}: check parentheses/brackets")
            elif "no member named" in msg or "has no member" in msg:
                result.suggestions.append(f"Type/struct member does not exist. Check field name at line {result.error_line}")
            elif "cannot convert" in msg or "incompatible" in msg:
                result.suggestions.append(f"Type mismatch at line {result.error_line}: check variable types")
            else:
                result.suggestions.append(f"Fix the error at {result.error_file}:{result.error_line}")
            
            break  # Focus on first error
        
        return result
    
    def analyze_test_failure(self, test_output: str) -> DebugResult:
        """Analyze test failure output."""
        result = self.DebugResult(status=self.STATUS_TEST_FAIL)
        
        # Python pytest/unittest format
        for m in re.finditer(r'(?:FAILED|Error|AssertionError)[:\s]+(.+)', test_output):
            result.diagnosis = m.group(1)
            break
        
        # Extract file references
        for m in re.finditer(r'File\s+"([^"]+)",\s+line\s+(\d+)', test_output):
            result.error_file = m.group(1)
            result.error_line = int(m.group(2))
            break
        
        # Extract stack trace
        trace_lines = []
        in_trace = False
        for line in test_output.splitlines():
            if "Traceback (most recent call last)" in line:
                in_trace = True
                continue
            if in_trace:
                if line.strip().startswith("File "):
                    trace_lines.append(line.strip())
                elif line.strip() and not line.startswith(" "):
                    if trace_lines:
                        result.stack_trace = "\n".join(trace_lines)
                    break
        
        # Suggest fix
        if "AssertionError" in test_output:
            result.suggestions.append("Check the assertion condition and expected vs actual values")
        if "ImportError" in test_output or "ModuleNotFoundError" in test_output:
            result.suggestions.append("Missing import or dependency. Check requirements.txt")
        if "TypeError" in test_output:
            result.suggestions.append(f"Type mismatch in function call at line {result.error_line}")
        if "AttributeError" in test_output:
            result.suggestions.append(f"Object does not have the requested attribute. Check variable type.")
        if "NameError" in test_output:
            result.suggestions.append(f"Undefined variable or function. Check spelling and imports.")
        
        return result
    
    def analyze_runtime_error(self, stderr: str) -> DebugResult:
        """Analyze runtime error output."""
        result = self.DebugResult(status=self.STATUS_RUNTIME_ERROR)
        result.diagnosis = stderr.strip()[:500]
        
        for m in re.finditer(r'Error:\s*(.+)', stderr):
            result.diagnosis = m.group(1)
            break
        
        # File/line extraction
        for m in re.finditer(r'at\s+(.+):(\d+)', stderr):
            result.error_file = m.group(1)
            result.error_line = int(m.group(2))
        
        return result
    
    def run_build(self, build_cmd: str, cwd: str = ".") -> DebugResult:
        """Run build command and analyze if it fails."""
        try:
            result = subprocess.run(
                build_cmd, shell=True, capture_output=True, text=True,
                timeout=120, cwd=cwd,
            )
            if result.returncode == 0:
                return self.DebugResult(status=self.STATUS_OK, diagnosis="Build succeeded")
            return self.analyze_build_error(result.stdout + result.stderr)
        except subprocess.TimeoutExpired:
            return self.DebugResult(
                status=self.STATUS_BUILD_FAIL,
                diagnosis="Build timed out (>120s). Check for infinite loops or large compilations.",
                suggestions=["Reduce compilation scope", "Check for circular dependencies"],
            )
    
    def run_tests(self, test_cmd: str, cwd: str = ".") -> DebugResult:
        """Run tests and analyze failures."""
        try:
            result = subprocess.run(
                test_cmd, shell=True, capture_output=True, text=True,
                timeout=120, cwd=cwd,
            )
            if result.returncode == 0:
                return self.DebugResult(status=self.STATUS_OK, diagnosis="All tests passed")
            return self.analyze_test_failure(result.stdout + result.stderr)
        except subprocess.TimeoutExpired:
            return self.DebugResult(
                status=self.STATUS_TEST_FAIL,
                diagnosis="Tests timed out (>120s). Check for infinite loops.",
                suggestions=["Add timeout to individual tests", "Check for deadlocks"],
            )
