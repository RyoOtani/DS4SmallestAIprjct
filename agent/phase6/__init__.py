"""
Phase 6: AI Software Engineer Professional

Modules:
- code_understanding  — AST parsing, symbol resolution, call graph, dependency graph
- architect           — Architecture analysis, pattern detection, improvement suggestions
- tool_calling        — Structured tool calling (OpenAI FC compatible), JSON Schema validation
- quality             — Quality pipeline: linter, formatter, type checker, static analysis
- critic              — Automated code review + debugger agent
- orchestrator        — End-to-end pipeline orchestrator
"""

from .code_understanding import (
    RepositoryAnalyzer, ASTParser, PythonParser, CppParser, RustParser,
    GoParser, JSParser, Symbol, CallEdge, Dependency,
)
from .architect import ArchitectAgent, Architecture, Component
from .tool_calling import ToolRegistry, ToolDefinition, ToolParam, ParamType, ToolCallParser
from .quality import QualityPipeline, Linter, Formatter, TypeChecker, StaticAnalyzer
from .critic import CriticAgent, DebuggerAgent, CodeReview, ReviewComment
from .orchestrator import Phase6Orchestrator

__all__ = [
    "RepositoryAnalyzer", "ASTParser", "PythonParser", "CppParser", "RustParser",
    "GoParser", "JSParser", "Symbol", "CallEdge", "Dependency",
    "ArchitectAgent", "Architecture", "Component",
    "ToolRegistry", "ToolDefinition", "ToolParam", "ParamType", "ToolCallParser",
    "QualityPipeline", "Linter", "Formatter", "TypeChecker", "StaticAnalyzer",
    "CriticAgent", "DebuggerAgent", "CodeReview", "ReviewComment",
    "Phase6Orchestrator",
]
