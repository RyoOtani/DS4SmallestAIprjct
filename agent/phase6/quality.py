"""
Phase 6: Quality Pipeline — Linter, Formatter, Type Checker, Static Analysis.

Automated code quality assurance with multi-language support.
"""

from __future__ import annotations
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class QualityIssue:
    file: str
    line: int = 0
    column: int = 0
    severity: str = "warning"  # error, warning, info
    rule: str = ""
    message: str = ""
    suggestion: str = ""


@dataclass
class QualityReport:
    issues: list[QualityIssue] = field(default_factory=list)
    total_errors: int = 0
    total_warnings: int = 0
    files_checked: int = 0
    duration_ms: float = 0.0
    
    @property
    def passed(self) -> bool:
        return self.total_errors == 0


class Linter:
    """Code linter with multi-language support."""
    
    LANG_CONFIG = {
        "python": {
            "cmd": ["flake8", "--select=E,F,W", "--max-line-length=120"],
            "parser": "flake8",
        },
        "c": {
            "cmd": ["clang-tidy", "--quiet"],
            "parser": "clang",
        },
        "cpp": {
            "cmd": ["clang-tidy", "--quiet"],
            "parser": "clang",
        },
        "javascript": {
            "cmd": ["npx", "eslint", "--format=json"],
            "parser": "eslint",
        },
        "typescript": {
            "cmd": ["npx", "eslint", "--format=json"],
            "parser": "eslint",
        },
    }
    
    def __init__(self, language: str = "auto"):
        self.language = language
    
    def lint_file(self, filepath: str) -> QualityReport:
        """Lint a single file."""
        lang = self._detect_lang(filepath)
        config = self.LANG_CONFIG.get(lang, {})
        
        if not config:
            report = QualityReport(files_checked=1)
            report.issues.append(QualityIssue(
                file=filepath, severity="info", rule="no-linter",
                message=f"No linter configured for {lang}",
            ))
            return report
        
        # Try to run the linter
        try:
            result = subprocess.run(
                config["cmd"] + [filepath],
                capture_output=True, text=True, timeout=30,
            )
            return self._parse_output(config["parser"], result.stdout, result.stderr, filepath)
        except FileNotFoundError:
            report = QualityReport(files_checked=1)
            report.issues.append(QualityIssue(
                file=filepath, severity="info", rule="linter-not-found",
                message=f"Linter '{config['cmd'][0]}' not installed. Install it for {lang} linting.",
            ))
            return report
        except subprocess.TimeoutExpired:
            report = QualityReport(files_checked=1)
            report.issues.append(QualityIssue(
                file=filepath, severity="error", rule="timeout",
                message="Linter timed out.",
            ))
            return report
    
    def lint_directory(self, path: str, extensions: Optional[list[str]] = None) -> QualityReport:
        """Lint all files in a directory."""
        if extensions is None:
            extensions = ['.py', '.c', '.h', '.cpp', '.js', '.ts']
        
        report = QualityReport()
        root = Path(path)
        
        skip_dirs = {'.git', '__pycache__', 'node_modules', 'build', 'dist', '.venv'}
        files = []
        for ext in extensions:
            for f in root.rglob(f'*{ext}'):
                if not any(s in f.parts for s in skip_dirs):
                    files.append(str(f))
        
        for f in files:
            r = self.lint_file(f)
            report.issues.extend(r.issues)
            report.total_errors += r.total_errors
            report.total_warnings += r.total_warnings
            report.files_checked += 1
        
        return report
    
    @staticmethod
    def _parse_output(parser: str, stdout: str, stderr: str, filepath: str) -> QualityReport:
        report = QualityReport(files_checked=1)
        
        if parser == "flake8":
            for line in stdout.splitlines():
                m = re.match(r'(.+):(\d+):(\d+):\s+(\w+)\s+(.+)', line)
                if m:
                    is_error = m.group(4).startswith('E')
                    report.issues.append(QualityIssue(
                        file=m.group(1), line=int(m.group(2)), column=int(m.group(3)),
                        severity="error" if is_error else "warning",
                        rule=m.group(4), message=m.group(5),
                    ))
                    if is_error:
                        report.total_errors += 1
                    else:
                        report.total_warnings += 1
        
        elif parser == "clang":
            for line in (stdout + stderr).splitlines():
                m = re.match(r'(.+):(\d+):(\d+):\s+(error|warning|note):\s+(.+)', line)
                if m:
                    sev = m.group(3)
                    report.issues.append(QualityIssue(
                        file=m.group(1), line=int(m.group(2)), column=int(m.group(3)),
                        severity=sev, message=m.group(4),
                    ))
                    if sev == "error":
                        report.total_errors += 1
                    elif sev == "warning":
                        report.total_warnings += 1
        
        elif parser == "eslint":
            try:
                data = json.loads(stdout)
                for item in data:
                    for msg in item.get("messages", []):
                        report.issues.append(QualityIssue(
                            file=item.get("filePath", filepath),
                            line=msg.get("line", 0), column=msg.get("column", 0),
                            severity="error" if msg.get("severity", 1) >= 2 else "warning",
                            rule=msg.get("ruleId", ""),
                            message=msg.get("message", ""),
                            suggestion=msg.get("suggestion", ""),
                        ))
                        if msg.get("severity", 1) >= 2:
                            report.total_errors += 1
                        else:
                            report.total_warnings += 1
            except json.JSONDecodeError:
                pass
        
        else:
            # Generic fallback: treat stderr as warnings
            for line in stderr.splitlines():
                if line.strip():
                    report.issues.append(QualityIssue(
                        file=filepath, severity="warning", message=line,
                    ))
                    report.total_warnings += 1
        
        return report
    
    @staticmethod
    def _detect_lang(filepath: str) -> str:
        ext = Path(filepath).suffix.lower()
        lang_map = {
            '.py': 'python', '.c': 'c', '.h': 'c',
            '.cpp': 'cpp', '.cc': 'cpp', '.hpp': 'cpp',
            '.js': 'javascript', '.jsx': 'javascript',
            '.ts': 'typescript', '.tsx': 'typescript',
        }
        return lang_map.get(ext, 'unknown')


class Formatter:
    """Code formatter with multi-language support."""
    
    FORMATTERS = {
        "python": ["black", "--quiet"],
        "c": ["clang-format", "-i"],
        "cpp": ["clang-format", "-i"],
        "javascript": ["npx", "prettier", "--write"],
        "typescript": ["npx", "prettier", "--write"],
    }
    
    def format_file(self, filepath: str, check_only: bool = False) -> dict:
        """Format a file. Returns {formatted: bool, output: str}."""
        lang = Linter._detect_lang(filepath)
        cmd = self.FORMATTERS.get(lang, [])
        
        if not cmd:
            return {"formatted": False, "error": f"No formatter for {lang}"}
        
        try:
            args = list(cmd)
            if check_only and cmd[0] in ("black", "clang-format"):
                args.append("--check" if cmd[0] == "black" else "--dry-run")
            args.append(filepath)
            
            result = subprocess.run(args, capture_output=True, text=True, timeout=30)
            return {
                "formatted": result.returncode == 0,
                "output": result.stdout + result.stderr,
            }
        except FileNotFoundError:
            return {"formatted": False, "error": f"Formatter '{cmd[0]}' not installed"}
        except subprocess.TimeoutExpired:
            return {"formatted": False, "error": "Formatter timed out"}


class TypeChecker:
    """Type checker for statically-typed languages."""
    
    CHECKERS = {
        "python": ["mypy", "--ignore-missing-imports"],
        "typescript": ["npx", "tsc", "--noEmit"],
    }
    
    def check_file(self, filepath: str) -> QualityReport:
        """Run type checking on a file."""
        lang = Linter._detect_lang(filepath)
        cmd = self.CHECKERS.get(lang, [])
        
        report = QualityReport(files_checked=1)
        if not cmd:
            report.issues.append(QualityIssue(
                file=filepath, severity="info", rule="no-type-checker",
                message=f"No type checker for {lang}",
            ))
            return report
        
        try:
            result = subprocess.run(
                cmd + [filepath], capture_output=True, text=True, timeout=60,
            )
            for line in (result.stdout + result.stderr).splitlines():
                m = re.match(r'(.+):(\d+):\s+(error|warning|note):\s+(.+)', line)
                if m:
                    is_error = m.group(3) == "error"
                    report.issues.append(QualityIssue(
                        file=m.group(1), line=int(m.group(2)),
                        severity="error" if is_error else "warning",
                        message=m.group(4),
                    ))
                    if is_error:
                        report.total_errors += 1
                    else:
                        report.total_warnings += 1
        except FileNotFoundError:
            report.issues.append(QualityIssue(
                file=filepath, severity="info", rule="type-checker-not-found",
                message=f"Type checker '{cmd[0]}' not installed.",
            ))
        
        return report


class StaticAnalyzer:
    """Static analysis for security and performance issues."""
    
    SECRET_PATTERNS = [
        (r'(?:api[_-]?key|apikey|secret|password|token|auth)\s*[:=]\s*["\'][\w\-\.]{20,}["\']', "hardcoded-secret"),
        (r'-----BEGIN\s+(?:RSA|EC|DSA|OPENSSH)\s+PRIVATE\s+KEY-----', "private-key"),
        (r'(?:TODO|FIXME|HACK|XXX)\s*:', "todo-marker"),
        (r'eval\s*\(', "dangerous-eval"),
        (r'exec\s*\(', "dangerous-exec"),
        (r'os\.system\s*\(', "dangerous-system-call"),
        (r'subprocess\.(?:call|run|Popen)\s*\(\s*["\']?\w+', "subprocess-usage"),
    ]
    
    def analyze_file(self, filepath: str) -> QualityReport:
        """Run static analysis on a file."""
        report = QualityReport(files_checked=1)
        
        try:
            source = Path(filepath).read_text(encoding="utf-8")
        except Exception:
            return report
        
        for pattern, rule in self.SECRET_PATTERNS:
            for m in re.finditer(pattern, source, re.IGNORECASE):
                line_no = source[:m.start()].count('\n') + 1
                report.issues.append(QualityIssue(
                    file=filepath, line=line_no,
                    severity="warning", rule=rule,
                    message=f"Detected: {rule}",
                ))
                report.total_warnings += 1
        
        # File size check
        lines = source.count('\n')
        if lines > 500:
            report.issues.append(QualityIssue(
                file=filepath, severity="info", rule="large-file",
                message=f"File has {lines} lines — consider splitting.",
                suggestion="Split into multiple modules by responsibility.",
            ))
        
        # Function complexity estimate
        if Linter._detect_lang(filepath) == 'python':
            report = self._py_complexity_check(source, filepath, report)
        
        return report
    
    def _py_complexity_check(self, source: str, filepath: str, report: QualityReport) -> QualityReport:
        import ast as py_ast
        try:
            tree = py_ast.parse(source)
            for node in py_ast.walk(tree):
                if isinstance(node, (py_ast.FunctionDef, py_ast.AsyncFunctionDef)):
                    # Count statements
                    stmt_count = sum(1 for _ in py_ast.walk(node) if isinstance(_, py_ast.stmt))
                    if stmt_count > 50:
                        report.issues.append(QualityIssue(
                            file=filepath, line=node.lineno,
                            severity="info", rule="complex-function",
                            message=f"Function '{node.name}' has ~{stmt_count} statements — consider refactoring.",
                        ))
        except SyntaxError:
            pass
        return report


class QualityPipeline:
    """Orchestrates full quality pipeline: Lint → Format → TypeCheck → Static Analysis."""
    
    def __init__(self):
        self.linter = Linter()
        self.formatter = Formatter()
        self.type_checker = TypeChecker()
        self.static_analyzer = StaticAnalyzer()
    
    def run_full_pipeline(self, path: str, auto_fix: bool = False) -> dict:
        """Run the complete quality pipeline and return a summary."""
        results = {
            "lint": None,
            "format": None,
            "type_check": None,
            "static_analysis": None,
            "overall_passed": True,
        }
        
        # 1. Lint
        lint_report = self.linter.lint_directory(path)
        results["lint"] = {
            "errors": lint_report.total_errors,
            "warnings": lint_report.total_warnings,
            "files": lint_report.files_checked,
            "passed": lint_report.passed,
        }
        
        # 2. Format
        if auto_fix:
            format_results = []
            root = Path(path)
            skip_dirs = {'.git', '__pycache__', 'node_modules', 'build', 'dist'}
            for ext in ('.py', '.c', '.h', '.cpp', '.js', '.ts'):
                for f in root.rglob(f'*{ext}'):
                    if not any(s in f.parts for s in skip_dirs):
                        format_results.append(self.formatter.format_file(str(f)))
            results["format"] = {"files_formatted": sum(1 for r in format_results if r.get("formatted"))}
        
        # 3. Type Check
        type_issues = QualityReport()
        root = Path(path)
        for f in root.rglob('*.py'):
            if '__pycache__' not in f.parts:
                r = self.type_checker.check_file(str(f))
                type_issues.issues.extend(r.issues)
                type_issues.total_errors += r.total_errors
        results["type_check"] = {
            "errors": type_issues.total_errors,
            "passed": type_issues.passed,
        }
        
        # 4. Static Analysis
        static_report = QualityReport()
        root = Path(path)
        skip_dirs = {'.git', '__pycache__', 'node_modules', 'build', 'dist', '.venv'}
        for ext in ('.py', '.c', '.h', '.cpp', '.js', '.ts', '.jsx', '.tsx'):
            for f in root.rglob(f'*{ext}'):
                if not any(s in f.parts for s in skip_dirs):
                    r = self.static_analyzer.analyze_file(str(f))
                    static_report.issues.extend(r.issues)
                    static_report.total_errors += r.total_errors
                    static_report.total_warnings += r.total_warnings
        results["static_analysis"] = {
            "errors": static_report.total_errors,
            "warnings": static_report.total_warnings,
            "passed": static_report.total_errors == 0,
        }
        
        # Overall
        results["overall_passed"] = (
            results["lint"]["passed"] and
            results["type_check"]["passed"] and
            results["static_analysis"]["passed"]
        )
        
        return results
