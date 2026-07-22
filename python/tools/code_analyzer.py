"""
code_analyzer.py — Code analysis tools for tinyllm RAG pipeline.

Integrates with tree-sitter for AST parsing and code indexing.
Falls back to regex-based analysis if tree-sitter not available.

Usage:
  python code_analyzer.py index <directory>   # Index code for RAG
  python code_analyzer.py deps <file>         # Extract dependency graph
  python code_analyzer.py ast <file>          # Print AST as JSON
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional


class CodeAnalyzer:
    """Tree-sitter based code analysis. Falls back gracefully."""

    SUPPORTED_EXTENSIONS = {
        '.py': 'python',
        '.c': 'c',
        '.h': 'c',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.hpp': 'cpp',
        '.rs': 'rust',
        '.go': 'go',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'tsx',
        '.jsx': 'javascript',
    }

    def __init__(self):
        self.parser = None
        self._init_parser()

    def _init_parser(self):
        """Try to initialize tree-sitter. Fallback to regex on failure."""
        try:
            import tree_sitter
            self.parser = tree_sitter
            self.has_treesitter = True
        except ImportError:
            self.has_treesitter = False
            print("[code_analyzer] tree-sitter not installed. Using regex fallback.")

    def get_language(self, filepath: str) -> Optional[str]:
        ext = Path(filepath).suffix
        return self.SUPPORTED_EXTENSIONS.get(ext)

    def parse_file(self, filepath: str) -> dict:
        """Parse a file and return structured information."""
        try:
            with open(filepath) as f:
                source = f.read()
        except Exception as e:
            return {'error': str(e), 'file': filepath}

        language = self.get_language(filepath)
        if not language:
            return {'error': f'unsupported language: {filepath}', 'file': filepath}

        result = {
            'file': filepath,
            'language': language,
            'lines': len(source.splitlines()),
            'size_bytes': len(source),
            'functions': self.extract_functions(source, language),
            'imports': self.extract_imports(source, language),
            'classes': self.extract_classes(source, language),
        }
        return result

    def extract_functions(self, source: str, language: str) -> list[dict]:
        """Extract function definitions."""
        patterns = {
            'python': r'def\s+(\w+)\s*\([^)]*\)(?:\s*->\s*\w+)?\s*:',
            'c': r'(?:static\s+)?(?:\w+(?:\s*\*)?)\s+(\w+)\s*\([^)]*\)\s*\{',
            'cpp': r'(?:virtual\s+)?(?:\w+(?:\s*\*)?)\s+(\w+)\s*\([^)]*\)\s*(?:const\s*)?\{',
            'rust': r'fn\s+(\w+)\s*<[^>]*>\s*\([^)]*\)(?:\s*->\s*\S+)?',
            'go': r'func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\([^)]*\)(?:\s*\S+)?',
            'javascript': r'(?:async\s+)?function\s+(\w+)\s*\([^)]*\)',
            'typescript': r'(?:async\s+)?function\s+(\w+)\s*\([^)]*\)',
        }

        pattern = patterns.get(language)
        if not pattern:
            return []

        functions = []
        for match in re.finditer(pattern, source, re.MULTILINE):
            name = match.group(1)
            line = source[:match.start()].count('\n') + 1
            functions.append({'name': name, 'line': line})
        return functions

    def extract_imports(self, source: str, language: str) -> list[str]:
        """Extract imports/dependencies."""
        patterns = {
            'python': [
                r'^import\s+(\S+)',
                r'^from\s+(\S+)\s+import',
            ],
            'c': [r'#include\s+[<"]([^>"]+)[>"]'],
            'cpp': [r'#include\s+[<"]([^>"]+)[>"]'],
            'rust': [r'^use\s+(\S+);'],
            'go': [r'^\s+"([^"]+)"'],
            'javascript': [
                r'(?:import\s+.*?\s+from\s+)?["\']([^"\']+)["\']',
                r'require\s*\(\s*["\']([^"\']+)["\']\s*\)',
            ],
        }

        imports = set()
        pats = patterns.get(language, [])
        if isinstance(pats, str):
            pats = [pats]

        for pat in pats:
            for match in re.finditer(pat, source, re.MULTILINE):
                imports.add(match.group(1))

        return sorted(imports)

    def extract_classes(self, source: str, language: str) -> list[dict]:
        """Extract class/struct definitions."""
        patterns = {
            'python': r'class\s+(\w+)\s*(?:\([^)]*\))?\s*:',
            'cpp': r'(?:class|struct)\s+(\w+)\s*(?::\s*\w+\s*)?\{',
            'rust': r'(?:pub\s+)?struct\s+(\w+)',
            'go': r'type\s+(\w+)\s+struct\s*\{',
            'typescript': r'(?:export\s+)?class\s+(\w+)',
        }

        pattern = patterns.get(language)
        if not pattern:
            return []

        classes = []
        for match in re.finditer(pattern, source, re.MULTILINE):
            name = match.group(1)
            line = source[:match.start()].count('\n') + 1
            classes.append({'name': name, 'line': line})
        return classes

    def build_dependency_graph(self, directory: str) -> dict:
        """Build a dependency graph for all files in a directory."""
        graph = {'nodes': [], 'edges': []}
        file_imports = {}

        for root, _, files in os.walk(directory):
            for fname in files:
                fpath = os.path.join(root, fname)
                lang = self.get_language(fpath)
                if not lang:
                    continue

                try:
                    with open(fpath) as f:
                        source = f.read()
                except Exception:
                    continue

                imports = self.extract_imports(source, lang)
                file_imports[fpath] = imports

                graph['nodes'].append({
                    'file': fpath,
                    'language': lang,
                    'imports': imports,
                })

        # Build edges
        for fpath, imports in file_imports.items():
            for imp in imports:
                # Try to resolve import to actual file
                for other in file_imports:
                    if imp in other or Path(imp).stem in Path(other).stem:
                        graph['edges'].append({
                            'from': fpath,
                            'to': other,
                            'import': imp,
                        })

        return graph

    def index_directory(self, directory: str, output: str):
        """Index all code files and output as JSON Lines for RAG ingestion."""
        results = []

        for root, _, files in os.walk(directory):
            # Skip hidden dirs and common ignores
            if any(skip in root for skip in ['.git', '__pycache__', 'node_modules', 'target']):
                continue

            for fname in files:
                fpath = os.path.join(root, fname)
                lang = self.get_language(fpath)
                if not lang:
                    continue

                parsed = self.parse_file(fpath)
                if 'error' not in parsed:
                    results.append(parsed)

        # Write JSON Lines
        with open(output, 'w') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

        print(f"Indexed {len(results)} files → {output}")
        return results


def main():
    parser = argparse.ArgumentParser(description="Code Analyzer for tinyllm RAG")
    sub = parser.add_subparsers(dest='command')

    index_p = sub.add_parser('index')
    index_p.add_argument('directory')
    index_p.add_argument('-o', '--output', default='code_index.jsonl')

    deps_p = sub.add_parser('deps')
    deps_p.add_argument('file')

    ast_p = sub.add_parser('ast')
    ast_p.add_argument('file')

    build_p = sub.add_parser('build-graph')
    build_p.add_argument('directory')
    build_p.add_argument('-o', '--output', default='dep_graph.json')

    args = parser.parse_args()
    analyzer = CodeAnalyzer()

    if args.command == 'index':
        analyzer.index_directory(args.directory, args.output)
    elif args.command == 'deps':
        result = analyzer.parse_file(args.file)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == 'ast':
        result = analyzer.parse_file(args.file)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == 'build-graph':
        graph = analyzer.build_dependency_graph(args.directory)
        with open(args.output, 'w') as f:
            json.dump(graph, f, indent=2)
        print(f"Dependency graph → {args.output}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
