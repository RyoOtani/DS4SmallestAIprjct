"""
Phase 6: Deep Code Understanding — AST, Symbol, Call Graph, Dependency Graph.

Provides multi-language structural code analysis beyond text search.
Supports Python, C, C++, JavaScript, TypeScript, Rust, Go.
"""

from __future__ import annotations
import ast as py_ast
import json
import os
import re
from collections import defaultdict
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Symbol:
    name: str
    kind: str  # function, class, method, variable, import, module
    file: str
    line: int
    signature: str = ""
    docstring: str = ""
    return_type: str = ""       # inferred or declared return type
    param_types: dict = field(default_factory=dict)  # param_name → type_string
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    visibility: str = "public"


@dataclass
class TypeHint:
    """Inferred type information for a variable or expression."""
    name: str
    type_str: str
    confidence: float = 0.5  # 0.0=guess, 1.0=declared
    source: str = "inferred"  # declared, inferred, heuristic


@dataclass
class CallEdge:
    caller: str
    callee: str
    caller_file: str
    callee_file: str = ""
    callee_line: int = 0
    arg_count: int = 0


@dataclass
class Dependency:
    source_file: str
    target_file: str
    kind: str  # import, include, require, call, inherit
    target_name: str = ""


class ASTParser(ABC):
    """Base AST parser with language-specific backends."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.source = ""
        self.symbols: list[Symbol] = []
        self.call_graph: list[CallEdge] = []
        self.imports: list[str] = []
        self._load()
    
    def _load(self):
        """Load source file with proper error handling."""
        try:
            filepath = Path(self.filepath)
            if not filepath.exists():
                self._load_error = f"File not found: {self.filepath}"
                return
            if filepath.stat().st_size > 10_000_000:  # 10 MB limit
                self._load_error = f"File too large: {self.filepath} ({filepath.stat().st_size} bytes)"
                return
            # Check for binary
            with open(filepath, 'rb') as f:
                head = f.read(512)
                if b'\x00' in head:
                    self._load_error = f"Binary file: {self.filepath}"
                    return
            self.source = filepath.read_text(encoding="utf-8", errors="replace")
            self._load_error = None
        except PermissionError:
            self._load_error = f"Permission denied: {self.filepath}"
            self.source = ""
        except Exception as e:
            self._load_error = f"Load error: {e}"
            self.source = ""
    
    @staticmethod
    def for_file(filepath: str) -> Optional[ASTParser]:
        ext = Path(filepath).suffix.lower()
        parsers = {
            '.py': PythonParser,
            '.c': CppParser,
            '.h': CppParser,
            '.cpp': CppParser,
            '.cc': CppParser,
            '.hpp': CppParser,
            '.rs': RustParser,
            '.go': GoParser,
            '.js': JSParser,
            '.ts': JSParser,
            '.jsx': JSParser,
            '.tsx': JSParser,
        }
        cls = parsers.get(ext)
        return cls(filepath) if cls else None
    
    def parse(self) -> ASTParser:
        raise NotImplementedError
    
    def find_symbol(self, name: str) -> Optional[Symbol]:
        for s in self.symbols:
            if s.name == name:
                return s
        return None
    
    def symbols_by_kind(self, kind: str) -> list[Symbol]:
        return [s for s in self.symbols if s.kind == kind]


class PythonParser(ASTParser):
    """Python AST parser using built-in `ast` module."""
    
    def parse(self) -> PythonParser:
        if not self.source.strip():
            return self
        
        try:
            tree = py_ast.parse(self.source, filename=self.filepath)
            self._walk(tree)
        except SyntaxError:
            pass
        return self
    
    def _walk(self, tree):
        for node in py_ast.walk(tree):
            if isinstance(node, py_ast.Import):
                for alias in node.names:
                    self.imports.append(alias.name)
                    self.symbols.append(Symbol(
                        name=alias.name, kind="import",
                        file=self.filepath, line=node.lineno,
                    ))
            elif isinstance(node, py_ast.ImportFrom):
                mod = node.module or ""
                for alias in node.names:
                    self.imports.append(f"{mod}.{alias.name}")
            elif isinstance(node, py_ast.FunctionDef):
                sig = self._fn_signature(node)
                calls = self._find_calls(node, node.name)
                self.symbols.append(Symbol(
                    name=node.name, kind="function", file=self.filepath,
                    line=node.lineno, signature=sig,
                    decorators=[self._decorator_name(d) for d in node.decorator_list],
                    children=calls,
                    docstring=py_ast.get_docstring(node) or "",
                ))
            elif isinstance(node, py_ast.AsyncFunctionDef):
                sig = self._fn_signature(node)
                self.symbols.append(Symbol(
                    name=node.name, kind="function", file=self.filepath,
                    line=node.lineno, signature=sig,
                    children=self._find_calls(node, node.name),
                ))
            elif isinstance(node, py_ast.ClassDef):
                bases = [self._name_of(b) for b in node.bases]
                methods = []
                for item in node.body:
                    if isinstance(item, (py_ast.FunctionDef, py_ast.AsyncFunctionDef)):
                        methods.append(item.name)
                self.symbols.append(Symbol(
                    name=node.name, kind="class", file=self.filepath,
                    line=node.lineno, parents=bases, children=methods,
                    docstring=py_ast.get_docstring(node) or "",
                ))
    
    def _fn_signature(self, node) -> str:
        args = []
        for a in node.args.args:
            arg_str = a.arg
            if a.annotation:
                arg_str += f": {self._name_of(a.annotation)}"
            args.append(arg_str)
        returns = ""
        if node.returns:
            returns = f" -> {self._name_of(node.returns)}"
        return f"({', '.join(args)}){returns}"
    
    def _find_calls(self, node, caller_name) -> list[str]:
        """Find all function calls within a function body."""
        calls = set()
        for child in py_ast.walk(node):
            if isinstance(child, py_ast.Call):
                name = self._call_name(child.func)
                if name and name != caller_name:
                    calls.add(name)
                    self.call_graph.append(CallEdge(
                        caller=caller_name, callee=name,
                        caller_file=self.filepath,
                        arg_count=len(child.args) + len(child.keywords),
                    ))
        return sorted(calls)
    
    @staticmethod
    def _call_name(func) -> str:
        if isinstance(func, py_ast.Name):
            return func.id
        if isinstance(func, py_ast.Attribute):
            parts = []
            node = func
            while isinstance(node, py_ast.Attribute):
                parts.append(node.attr)
                node = node.value
            if isinstance(node, py_ast.Name):
                parts.append(node.id)
            return ".".join(reversed(parts))
        return ""
    
    @staticmethod
    def _name_of(node) -> str:
        if isinstance(node, py_ast.Name):
            return node.id
        if isinstance(node, py_ast.Attribute):
            return f"{PythonParser._name_of(node.value)}.{node.attr}"
        if isinstance(node, py_ast.Subscript):
            return f"{PythonParser._name_of(node.value)}[...]"
        if node is None:
            return "None"
        return "?"
    
    @staticmethod
    def _decorator_name(node) -> str:
        if isinstance(node, py_ast.Name):
            return node.id
        if isinstance(node, py_ast.Attribute):
            return PythonParser._name_of(node)
        if isinstance(node, py_ast.Call):
            return PythonParser._name_of(node.func)
        return "?"


class CppParser(ASTParser):
    """C/C++ parser using regex-based extraction (lightweight, no external deps)."""
    
    def parse(self) -> CppParser:
        if not self.source.strip():
            return self
        
        self._parse_imports()
        self._parse_functions()
        self._parse_classes()
        self._parse_calls()
        return self
    
    def _parse_imports(self):
        for m in re.finditer(r'#include\s+[<"]([^>"]+)[>"]', self.source):
            self.imports.append(m.group(1))
    
    def _parse_functions(self):
        pattern = r'''
            (?:static\s+|inline\s+|virtual\s+|explicit\s+|constexpr\s+)*
            ([\w:]+(?:<[^>]*>)?(?:\s*\*)?\s+){1,2}
            (\w+)\s*
            \((.*?)\)\s*
            (?:const\s*)?(?:override\s*)?(?:noexcept\s*)?
            (?:\{|;)
        '''
        for m in re.finditer(pattern, self.source, re.VERBOSE):
            name = m.group(2)
            if name in ('if', 'while', 'for', 'switch', 'return', 'throw', 'catch', 'sizeof', 'decltype'):
                continue
            args = m.group(3).strip()
            self.symbols.append(Symbol(
                name=name, kind="function", file=self.filepath,
                line=self.source[:m.start()].count('\n') + 1,
                signature=f"({args})",
            ))
    
    def _parse_classes(self):
        pattern = r'(?:class|struct)\s+(\w+)\s*(?::\s*([^{]+))?\s*\{'
        for m in re.finditer(pattern, self.source):
            name = m.group(1)
            bases = [b.strip() for b in (m.group(2) or "").split(",") if b.strip()] if m.group(2) else []
            self.symbols.append(Symbol(
                name=name, kind="class", file=self.filepath,
                line=self.source[:m.start()].count('\n') + 1,
                parents=bases,
            ))
    
    def _parse_calls(self):
        # Find all function calls
        for m in re.finditer(r'\b(\w+)\s*\(', self.source):
            name = m.group(1)
            if name in ('if', 'while', 'for', 'switch', 'return', 'sizeof', 'defined', '__LINE__', '__FILE__'):
                continue
            # Avoid matching function declarations
            if re.match(r'^[\w:]+\s+\w+\s*\(', self.source[max(0, m.start()-30):m.start()+len(name)+1]):
                continue
            self.call_graph.append(CallEdge(
                caller="(callsite)", callee=name,
                caller_file=self.filepath,
            ))


class RustParser(CppParser):
    """Rust parser (extends CppParser pattern)."""
    
    def parse(self) -> RustParser:
        if not self.source.strip():
            return self
        
        # Functions
        for m in re.finditer(r'(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\((.*?)\)\s*(?:->\s*\S+)?\s*\{', self.source):
            name = m.group(1)
            args = m.group(2).strip() if m.group(2) else ""
            self.symbols.append(Symbol(
                name=name, kind="function", file=self.filepath,
                line=self.source[:m.start()].count('\n') + 1,
                signature=f"({args})",
            ))
        
        # Structs / impl blocks
        for m in re.finditer(r'(?:pub\s+)?struct\s+(\w+)', self.source):
            self.symbols.append(Symbol(
                name=m.group(1), kind="class", file=self.filepath,
                line=self.source[:m.start()].count('\n') + 1,
            ))
        
        # Imports
        for m in re.finditer(r'^use\s+(\S+);', self.source, re.MULTILINE):
            self.imports.append(m.group(1))
        
        # Calls
        self._parse_calls()
        return self


class GoParser(CppParser):
    """Go parser."""
    
    def parse(self) -> GoParser:
        if not self.source.strip():
            return self
        
        for m in re.finditer(r'func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\((.*?)\)\s*(?:\S+)?\s*\{', self.source):
            self.symbols.append(Symbol(
                name=m.group(1), kind="function", file=self.filepath,
                line=self.source[:m.start()].count('\n') + 1,
            ))
        
        for m in re.finditer(r'type\s+(\w+)\s+struct\s*\{', self.source):
            self.symbols.append(Symbol(
                name=m.group(1), kind="class", file=self.filepath,
                line=self.source[:m.start()].count('\n') + 1,
            ))
        
        for m in re.finditer(r'^\s+"([^"]+)"', self.source, re.MULTILINE):
            self.imports.append(m.group(1))
        
        self._parse_calls()
        return self


class JSParser(CppParser):
    """JavaScript/TypeScript parser."""
    
    def parse(self) -> JSParser:
        if not self.source.strip():
            return self
        
        for m in re.finditer(r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*(?:<[^>]*>)?\s*\((.*?)\)', self.source):
            self.symbols.append(Symbol(
                name=m.group(1), kind="function", file=self.filepath,
                line=self.source[:m.start()].count('\n') + 1,
            ))
        
        for m in re.finditer(r'(?:export\s+)?(?:abstract\s+)?class\s+(\w+)', self.source):
            self.symbols.append(Symbol(
                name=m.group(1), kind="class", file=self.filepath,
                line=self.source[:m.start()].count('\n') + 1,
            ))
        
        for m in re.finditer(r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\((.*?)\)\s*=>', self.source):
            self.symbols.append(Symbol(
                name=m.group(1), kind="function", file=self.filepath,
                line=self.source[:m.start()].count('\n') + 1,
            ))
        
        for m in re.finditer(r'(?:import\s+.*?\s+from\s+)?["\']([^"\']+)["\']|require\s*\(\s*["\']([^"\']+)["\']\s*\)', self.source):
            imp = m.group(1) or m.group(2)
            if imp:
                self.imports.append(imp)
        
        self._parse_calls()
        return self


class RepositoryAnalyzer:
    """Analyzes entire repository: cross-file symbol resolution, call graph, deps."""
    
    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self.parsers: dict[str, ASTParser] = {}
        self.symbol_index: dict[str, list[Symbol]] = defaultdict(list)
        self.dep_graph: dict[str, list[Dependency]] = defaultdict(list)
        self.all_call_edges: list[CallEdge] = []
        self.metrics: dict = {}
    
    def scan(self, extensions: Optional[set[str]] = None) -> RepositoryAnalyzer:
        """Scan and parse all supported files in the repository."""
        if extensions is None:
            extensions = {'.py', '.c', '.h', '.cpp', '.cc', '.hpp', '.rs', '.go', '.js', '.ts', '.tsx', '.jsx'}
        
        files = []
        for ext in extensions:
            files.extend(self.root.rglob(f'*{ext}'))
        
        # Skip hidden dirs and common ignores
        skip_dirs = {'.git', '__pycache__', 'node_modules', 'target', 'build', 'dist', '.venv', 'venv'}
        files = [f for f in files if not any(d in f.parts for d in skip_dirs)]
        
        # Parse each file
        for fpath in files:
            parser = ASTParser.for_file(str(fpath))
            if parser:
                parser.parse()
                self.parsers[str(fpath)] = parser
                
                for sym in parser.symbols:
                    key = f"{sym.name}::{sym.kind}"
                    self.symbol_index[key].append(sym)
        
        # Build dependency graph
        self._build_dep_graph()
        
        # Compute metrics
        self._compute_metrics()
        
        return self
    
    def _build_dep_graph(self):
        """Cross-file dependency resolution."""
        for fpath, parser in self.parsers.items():
            for imp in parser.imports:
                # Try to resolve import to actual file
                for other_path in self.parsers:
                    if imp in other_path or Path(imp).stem in Path(other_path).stem:
                        self.dep_graph[fpath].append(Dependency(
                            source_file=fpath, target_file=other_path,
                            kind="import", target_name=imp,
                        ))
    
    def _compute_metrics(self):
        """Compute repository-level metrics."""
        total_files = len(self.parsers)
        total_symbols = sum(len(p.symbols) for p in self.parsers.values())
        total_functions = sum(len(p.symbols_by_kind("function")) for p in self.parsers.values())
        total_classes = sum(len(p.symbols_by_kind("class")) for p in self.parsers.values())
        
        # File complexity (simple: symbols per file)
        complexities = {f: len(p.symbols) for f, p in self.parsers.items()}
        
        self.metrics = {
            "total_files": total_files,
            "total_symbols": total_symbols,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "avg_symbols_per_file": total_symbols / max(total_files, 1),
            "most_complex_files": sorted(complexities.items(), key=lambda x: x[1], reverse=True)[:10],
        }
    
    def find_symbol_usages(self, name: str) -> list[Symbol]:
        """Find all definitions of a symbol across the repo."""
        results = []
        for key, syms in self.symbol_index.items():
            if key.startswith(f"{name}::"):
                results.extend(syms)
        return results
    
    def find_callers(self, callee_name: str) -> list[CallEdge]:
        """Find all callers of a given function."""
        edges = []
        for parser in self.parsers.values():
            for edge in parser.call_graph:
                if edge.callee == callee_name:
                    edges.append(edge)
        return edges
    
    def find_callees(self, caller_name: str) -> list[CallEdge]:
        """Find all functions called by a given function."""
        edges = []
        for parser in self.parsers.values():
            for edge in parser.call_graph:
                if edge.caller == caller_name:
                    edges.append(edge)
        return edges
    
    def dependency_chain(self, target_file: str) -> list[str]:
        """Find all files that depend on target_file (transitively)."""
        visited = set()
        queue = [target_file]
        while queue:
            f = queue.pop(0)
            if f in visited:
                continue
            visited.add(f)
            for dep in self.dep_graph.get(f, []):
                if dep.target_file not in visited:
                    queue.append(dep.target_file)
        return sorted(visited)
    
    def to_dict(self) -> dict:
        return {
            "metrics": self.metrics,
            "files": list(self.parsers.keys()),
            "dependency_graph": {
                f: [{"target": d.target_file, "kind": d.kind, "name": d.target_name}
                    for d in deps]
                for f, deps in self.dep_graph.items()
            },
        }
    
    def context_for_file(self, filepath: str, query: str = "", max_snippets: int = 5) -> str:
        """Build context string: file symbols + related files + relevant snippets."""
        lines = []
        parser = self.parsers.get(filepath)
        
        if parser:
            lines.append(f"=== {filepath} ===")
            lines.append(f"Symbols ({len(parser.symbols)}):")
            for sym in parser.symbols[:30]:
                type_info = ""
                if sym.return_type:
                    type_info = f" → {sym.return_type}"
                lines.append(f"  {sym.kind:10s} {sym.name:30s} line {sym.line}{type_info}")
            
            if parser.imports:
                lines.append(f"\nImports: {', '.join(parser.imports[:20])}")
        
        # Dependencies
        deps = self.dep_graph.get(filepath, [])
        if deps:
            lines.append(f"\n=== Dependencies ({len(deps)}) ===")
            for d in deps[:15]:
                lines.append(f"  → {d.target_file}  ({d.kind}: {d.target_name})")
        
        # Call graph
        edges = []
        if parser:
            edges = parser.call_graph[:20]
        if edges:
            lines.append(f"\n=== Call Graph ({len(edges)} edges) ===")
            for e in edges:
                lines.append(f"  {e.caller} → {e.callee}  ({e.caller_file})")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Mypy / Pyright Integration — external type checker for deep type info
# ═══════════════════════════════════════════════════════════════════════

class ExternalTypeChecker:
    """
    Runs external type checkers (mypy, pyright) and parses their output
    to enrich Symbol type information beyond AST-level inference.
    
    Supported checkers: mypy, pyright (pylance)
    """

    CHECKERS = {
        "mypy": ["mypy", "--no-error-summary", "--show-error-codes"],
        "pyright": ["pyright", "--outputjson"],
    }

    def __init__(self, checker: str = "mypy"):
        self.checker = checker
        self._available = self._check_available()

    def _check_available(self) -> bool:
        import shutil
        cmd = self.CHECKERS.get(self.checker, [])
        return bool(cmd and shutil.which(cmd[0]))

    def analyze_file(self, filepath: str) -> dict:
        """Run type checker on a file and parse type information."""
        if not self._available:
            return {"error": f"{self.checker} not installed", "types": []}

        import subprocess
        try:
            result = subprocess.run(
                self.CHECKERS[self.checker] + [filepath],
                capture_output=True, text=True, timeout=30,
            )
            return self._parse_output(result.stdout + result.stderr, filepath)
        except FileNotFoundError:
            return {"error": f"{self.checker} not found"}
        except subprocess.TimeoutExpired:
            return {"error": "Type check timed out"}

    def _parse_output(self, output: str, filepath: str) -> dict:
        """Parse mypy output for type annotations and issues.

        Mypy format:
          file:line: error: message  [error-code]
          file:line: note: message

        Pyright format (JSON):
          {"version": "...", "generalDiagnostics": [...], "diagnostics": [...]}
        """
        import json

        types = []
        notes = []

        if self.checker == "pyright":
            try:
                data = json.loads(output)
                for diag in data.get("diagnostics", []):
                    types.append({
                        "line": diag.get("range", {}).get("start", {}).get("line", 0) + 1,
                        "message": diag.get("message", ""),
                        "severity": diag.get("severity", "information"),
                        "rule": diag.get("rule", ""),
                    })
            except json.JSONDecodeError:
                pass
        else:
            # mypy text output
            import re
            for line in output.splitlines():
                m = re.match(
                    r'(.+):(\d+):\s+(error|warning|note):\s+(.+)',
                    line,
                )
                if m:
                    is_error = m.group(3) == "error"
                    category = "error" if is_error else "note"
                    types.append({
                        "file": m.group(1),
                        "line": int(m.group(2)),
                        "category": category,
                        "message": m.group(4),
                    })

        return {
            "file": filepath,
            "checker": self.checker,
            "issues": len(types),
            "types": types,
            "notes": notes,
        }

    def enrich_symbols(
        self, parser: "ASTParser", filepath: str,
    ) -> list[dict]:
        """Run type checker and update Symbol type information."""
        result = self.analyze_file(filepath)
        if "error" in result:
            return []

        enriched = []
        type_map: dict[str, str] = {}
        inferred_types: list[TypeHint] = []

        # Extract type hints from checker output
        for issue in result.get("types", []):
            msg = issue.get("message", "")
            line = issue.get("line", 0)

            # Pattern: 'Variable "x" is of type "int"'
            var_match = re.search(
                r'(?:Variable|Argument)\s+"?(\w+)"?\s+(?:is of type|has type)\s+"?([^"]+)"?', msg,
            )
            if var_match:
                name = var_match.group(1)
                type_str = var_match.group(2)
                type_map[name] = type_str
                inferred_types.append(TypeHint(
                    name=name, type_str=type_str,
                    confidence=0.9, source=f"{self.checker}",
                ))
                enriched.append({"name": name, "type": type_str, "line": line})

            # Pattern: 'Incompatible return type' → extract expected type
            return_match = re.search(
                r'(?:Incompatible return value type|Returning).*?"(?:[^"]*)"\s+instead of\s+"?([^"]+)"?', msg,
            )
            if return_match:
                ret_type = return_match.group(1)
                enriched.append({"return_type": ret_type, "line": line})

        # Update parser symbols with inferred types
        for sym in parser.symbols:
            if sym.name in type_map and not sym.return_type:
                sym.return_type = type_map[sym.name]
                sym.param_types = {"inferred_return": type_map[sym.name]}

        return enriched
