"""
Phase 6: Structured Tool Calling — OpenAI Function Calling compatible,
         JSON Schema-based tool definitions, type-safe argument validation.
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class ParamType(Enum):
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    FILE_CONTENT = "file_content"  # extended: inline file content


@dataclass
class ToolParam:
    name: str
    type: ParamType
    description: str = ""
    required: bool = True
    enum_values: Optional[list[str]] = None
    default: Any = None
    items_type: Optional[ParamType] = None  # for array items


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: list[ToolParam] = field(default_factory=list)
    category: str = "general"  # code, file, system, search, quality, git
    requires_sandbox: bool = False
    handler: Optional[Callable] = None  # actual Python callable
    
    def to_openai_schema(self) -> dict:
        """Convert to OpenAI Function Calling JSON Schema."""
        properties = {}
        required = []
        for p in self.parameters:
            prop = {
                "type": p.type.value,
                "description": p.description,
            }
            if p.enum_values:
                prop["enum"] = p.enum_values
            if p.default is not None:
                prop["default"] = p.default
            if p.type == ParamType.ARRAY and p.items_type:
                prop["items"] = {"type": p.items_type.value}
            properties[p.name] = prop
            if p.required:
                required.append(p.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class ToolRegistry:
    """Registry of available tools with schema validation."""
    
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._register_builtins()
    
    def register(self, tool: ToolDefinition) -> ToolRegistry:
        self._tools[tool.name] = tool
        return self
    
    def get(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)
    
    def list_all(self) -> list[ToolDefinition]:
        return list(self._tools.values())
    
    def list_by_category(self, category: str) -> list[ToolDefinition]:
        return [t for t in self._tools.values() if t.category == category]
    
    def to_openai_tools(self) -> list[dict]:
        return [t.to_openai_schema() for t in self._tools.values()]
    
    def validate_args(self, tool_name: str, args: dict) -> tuple[bool, Optional[str]]:
        """Validate arguments against tool schema. Returns (valid, error_msg)."""
        tool = self.get(tool_name)
        if not tool:
            return False, f"Unknown tool: {tool_name}"
        
        for param in tool.parameters:
            if param.required and param.name not in args:
                return False, f"Missing required parameter: {param.name}"
            
            if param.name in args:
                value = args[param.name]
                if not self._check_type(value, param.type, param.items_type):
                    return False, f"Parameter '{param.name}' expected {param.type.value}, got {type(value).__name__}"
        
        return True, None
    
    @staticmethod
    def _check_type(value: Any, expected: ParamType, items_type: Optional[ParamType] = None) -> bool:
        if expected == ParamType.STRING:
            return isinstance(value, str)
        if expected == ParamType.NUMBER:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected == ParamType.INTEGER:
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == ParamType.BOOLEAN:
            return isinstance(value, bool)
        if expected == ParamType.ARRAY:
            if not isinstance(value, list):
                return False
            if items_type:
                checker = lambda v: ToolRegistry._check_type(v, items_type)
                return all(checker(item) for item in value)
            return True
        if expected == ParamType.OBJECT:
            return isinstance(value, dict)
        if expected == ParamType.FILE_CONTENT:
            if not isinstance(value, str):
                return False
            if value.startswith(("http://", "https://", "ftp://")):
                return False  # reject URLs
            if len(value) > 1_000_000:
                return False  # 1 MB size limit
            return True
        return True
    
    def _register_builtins(self):
        """Register standard code tools with real handlers."""
        import subprocess, os
        
        # File operations
        self.register(ToolDefinition(
            name="read_file",
            description="Read the contents of a file.",
            category="file",
            parameters=[
                ToolParam("path", ParamType.STRING, "Path to the file"),
                ToolParam("start_line", ParamType.INTEGER, "Start line (1-indexed)", required=False, default=1),
                ToolParam("end_line", ParamType.INTEGER, "End line (inclusive)", required=False, default=0),
            ],
            handler=lambda path, start_line=1, end_line=0: _read_file(path, start_line, end_line),
        ))
        
        self.register(ToolDefinition(
            name="write_file",
            description="Write or overwrite a file with new content.",
            category="file",
            parameters=[
                ToolParam("path", ParamType.STRING, "Path to the file"),
                ToolParam("content", ParamType.STRING, "Content to write"),
            ],
            handler=lambda path, content: _write_file(path, content),
        ))
        
        self.register(ToolDefinition(
            name="edit_file",
            description="Apply a unified diff patch to a file.",
            category="file",
            parameters=[
                ToolParam("path", ParamType.STRING, "Path to the file to patch"),
                ToolParam("patch", ParamType.STRING, "Unified diff patch to apply"),
            ],
            handler=lambda path, patch: _apply_patch(path, patch),
        ))
        
        # Code search
        self.register(ToolDefinition(
            name="find_symbol",
            description="Find a symbol (function, class, variable) in the codebase.",
            category="code",
            parameters=[
                ToolParam("name", ParamType.STRING, "Symbol name to search for"),
                ToolParam("kind", ParamType.STRING, "Kind of symbol", required=False,
                          enum_values=["function", "class", "variable", "any"]),
            ],
            handler=lambda name, kind="any": _find_symbol(name, kind),
        ))
        
        self.register(ToolDefinition(
            name="search_code",
            description="Search for a pattern in the codebase.",
            category="code",
            parameters=[
                ToolParam("pattern", ParamType.STRING, "Search pattern (regex or text)"),
                ToolParam("file_pattern", ParamType.STRING, "File glob pattern", required=False, default="**/*"),
                ToolParam("max_results", ParamType.INTEGER, "Max results", required=False, default=20),
            ],
            handler=lambda pattern, file_pattern="**/*", max_results=20: _search_code(pattern, file_pattern, max_results),
        ))
        
        # System
        self.register(ToolDefinition(
            name="run_command",
            description="Execute a shell command and return output.",
            category="system",
            parameters=[
                ToolParam("command", ParamType.STRING, "Shell command to execute"),
                ToolParam("cwd", ParamType.STRING, "Working directory", required=False, default="."),
                ToolParam("timeout", ParamType.INTEGER, "Timeout in seconds", required=False, default=30),
            ],
            requires_sandbox=True,
            handler=lambda command, cwd=".", timeout=30: _run_command(command, cwd, timeout),
        ))
        
        self.register(ToolDefinition(
            name="run_tests",
            description="Run the test suite.",
            category="system",
            parameters=[
                ToolParam("test_path", ParamType.STRING, "Path to test file or directory", required=False),
                ToolParam("filter", ParamType.STRING, "Test name filter", required=False),
            ],
            requires_sandbox=True,
            handler=lambda test_path=None, filter=None: _run_tests(test_path, filter),
        ))
        
        self.register(ToolDefinition(
            name="find_callers",
            description="Find all functions that call a given function.",
            category="code",
            parameters=[
                ToolParam("function_name", ParamType.STRING, "Name of the function to find callers for"),
            ],
        ))
        
        self.register(ToolDefinition(
            name="search_code",
            description="Search for a pattern in the codebase.",
            category="code",
            parameters=[
                ToolParam("pattern", ParamType.STRING, "Search pattern (regex or text)"),
                ToolParam("file_pattern", ParamType.STRING, "File glob pattern", required=False, default="**/*"),
                ToolParam("max_results", ParamType.INTEGER, "Max results", required=False, default=20),
            ],
        ))
        
        # Build & Test
        self.register(ToolDefinition(
            name="run_command",
            description="Execute a shell command and return output.",
            category="system",
            parameters=[
                ToolParam("command", ParamType.STRING, "Shell command to execute"),
                ToolParam("cwd", ParamType.STRING, "Working directory", required=False, default="."),
                ToolParam("timeout", ParamType.INTEGER, "Timeout in seconds", required=False, default=30),
            ],
            requires_sandbox=True,
        ))
        
        self.register(ToolDefinition(
            name="run_tests",
            description="Run the test suite.",
            category="system",
            parameters=[
                ToolParam("test_path", ParamType.STRING, "Path to test file or directory", required=False),
                ToolParam("filter", ParamType.STRING, "Test name filter", required=False),
            ],
            requires_sandbox=True,
        ))
        
        self.register(ToolDefinition(
            name="run_linter",
            description="Run a code linter on specified files.",
            category="quality",
            parameters=[
                ToolParam("path", ParamType.STRING, "File or directory to lint"),
                ToolParam("language", ParamType.STRING, "Language for linting", required=False,
                          enum_values=["python", "c", "cpp", "javascript", "typescript", "auto"]),
            ],
        ))
        
        # Git operations
        self.register(ToolDefinition(
            name="git_diff",
            description="Show git diff (unstaged or staged changes).",
            category="git",
            parameters=[
                ToolParam("staged", ParamType.BOOLEAN, "Show staged changes only", required=False, default=False),
            ],
        ))
        
        self.register(ToolDefinition(
            name="git_checkpoint",
            description="Create a git commit checkpoint for rollback.",
            category="git",
            parameters=[
                ToolParam("message", ParamType.STRING, "Commit message"),
            ],
        ))
        
        self.register(ToolDefinition(
            name="git_rollback",
            description="Rollback to a previous git commit.",
            category="git",
            parameters=[
                ToolParam("commit", ParamType.STRING, "Commit hash or ref to rollback to", required=False, default="HEAD~1"),
            ],
        ))
        
        # Web / RAG
        self.register(ToolDefinition(
            name="rag_search",
            description="Search the local code index for relevant context.",
            category="search",
            parameters=[
                ToolParam("query", ParamType.STRING, "Search query"),
                ToolParam("max_results", ParamType.INTEGER, "Max results", required=False, default=8),
            ],
        ))
        
        self.register(ToolDefinition(
            name="web_search",
            description="Search the web for information (cached).",
            category="search",
            parameters=[
                ToolParam("query", ParamType.STRING, "Search query"),
            ],
        ))


class ToolCallParser:
    """Parses tool calls from LLM output (supports both XML and JSON formats)."""
    
    @staticmethod
    def parse_xml(text: str) -> list[dict]:
        """Parse <tool_call> XML format."""
        calls = []
        pattern = r'<tool_call>\s*<name>(.*?)</name>\s*<params>(.*?)</params>\s*</tool_call>'
        for m in re.finditer(pattern, text, re.DOTALL):
            name = m.group(1).strip()
            params_str = m.group(2).strip()
            try:
                params = json.loads(params_str)
            except json.JSONDecodeError:
                params = {"raw": params_str}
            calls.append({"name": name, "params": params})
        return calls
    
    @staticmethod
    def parse_openai(text: str) -> list[dict]:
        """Parse OpenAI Function Calling JSON format."""
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "name" in data:
                return [data]
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from text
        for m in re.finditer(r'\{[^{}]*"name"\s*:\s*"[^"]+"[^{}]*\}', text):
            try:
                return [json.loads(m.group(0))]
            except json.JSONDecodeError:
                continue
        return []
    
    @staticmethod
    def parse(text: str) -> list[dict]:
        """Auto-detect format and parse."""
        if '<tool_call>' in text:
            return ToolCallParser.parse_xml(text)
        if '"name"' in text:
            return ToolCallParser.parse_openai(text)
        return []


# ═══════════════════════════════════════════════════════════════
# Built-in Tool Handlers
# ═══════════════════════════════════════════════════════════════

def _read_file(path: str, start_line: int = 1, end_line: int = 0) -> str:
    """Read file contents, optionally with line range."""
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        if end_line <= 0:
            end_line = len(lines)
        start = max(0, start_line - 1)
        end = min(len(lines), end_line)
        return ''.join(lines[start:end])
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except Exception as e:
        return f"Error reading {path}: {e}"


def _write_file(path: str, content: str) -> str:
    """Write content to a file."""
    import os
    try:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing {path}: {e}"


def _apply_patch(path: str, patch: str) -> str:
    """Apply a unified diff patch."""
    import subprocess, tempfile, os
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
            f.write(patch)
            patch_file = f.name
        result = subprocess.run(
            ['patch', '-p1', '-i', patch_file, path],
            capture_output=True, text=True, timeout=30
        )
        os.unlink(patch_file)
        if result.returncode == 0:
            return f"Patch applied to {path}"
        return f"Patch failed: {result.stderr or result.stdout}"
    except Exception as e:
        return f"Error applying patch: {e}"


def _find_symbol(name: str, kind: str = "any") -> str:
    """Search for a symbol in the codebase using grep."""
    import subprocess
    try:
        patterns = {
            'function': rf'def\s+{name}\b|func\s+.*{name}\b|function\s+{name}\b',
            'class': rf'class\s+{name}\b',
            'variable': rf'\b{name}\s*=',
            'any': rf'\b{name}\b',
        }
        pattern = patterns.get(kind, patterns['any'])
        result = subprocess.run(
            ['grep', '-rn', '--include=*.py', '--include=*.js', '--include=*.ts',
             '--include=*.rs', '--include=*.go', '--include=*.c', '--include=*.h',
             '-E', pattern, '.'],
            capture_output=True, text=True, timeout=10
        )
        output = result.stdout.strip()
        return output if output else f"No matches found for '{name}'"
    except Exception as e:
        return f"Symbol search failed: {e}"


def _search_code(pattern: str, file_pattern: str = "**/*", max_results: int = 20) -> str:
    """Search codebase for a pattern."""
    import subprocess
    try:
        result = subprocess.run(
            ['grep', '-rn', '-m', str(max_results), pattern, '.'],
            capture_output=True, text=True, timeout=15
        )
        output = result.stdout.strip()
        return output[:5000] if output else f"No matches for '{pattern}'"
    except Exception as e:
        return f"Code search failed: {e}"


def _run_command(command: str, cwd: str = ".", timeout: int = 30) -> str:
    """Execute a command safely (shell=False, no injection risk)."""
    import subprocess, shlex
    try:
        cmd_parts = shlex.split(command)
        result = subprocess.run(
            cmd_parts, capture_output=True, text=True,
            cwd=cwd, timeout=timeout, shell=False,
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            output += f"\n[stderr]\n{result.stderr.strip()}"
        return output[:5000] if output else f"Command completed (exit {result.returncode})"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s"
    except Exception as e:
        return f"Command execution failed: {e}"


def _run_tests(test_path: str = None, filter: str = None) -> str:
    """Run Python tests."""
    import subprocess
    try:
        cmd = ['python', '-m', 'pytest', '-q']
        if test_path:
            cmd.append(test_path)
        if filter:
            cmd.extend(['-k', filter])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.stdout.strip()[-3000:] if result.stdout else f"Tests completed (exit {result.returncode})"
    except Exception as e:
        return f"Test execution failed: {e}"
