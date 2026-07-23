"""Tests for Phase 6: AI Software Engineer Professional."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.phase6.code_understanding import (
    RepositoryAnalyzer, PythonParser, CppParser, Symbol, CallEdge,
)
from agent.phase6.architect import ArchitectAgent, Architecture, Component
from agent.phase6.tool_calling import ToolRegistry, ToolCallParser, ParamType, ToolParam, ToolDefinition
from agent.phase6.quality import QualityPipeline, Linter, StaticAnalyzer
from agent.phase6.critic import CriticAgent, DebuggerAgent


def test_python_parser():
    """Test Python AST parser."""
    code = '''
import os
from pathlib import Path

class MyClass:
    """A test class."""
    def method_a(self, x: int) -> str:
        return str(x)

def helper_func():
    return MyClass().method_a(42)
'''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        f.flush()
        parser = PythonParser(f.name)
        parser.parse()
        os.unlink(f.name)
    
    assert len(parser.symbols) >= 3, f"Expected 3+ symbols, got {len(parser.symbols)}"
    
    func = parser.find_symbol("helper_func")
    assert func is not None, "helper_func not found"
    assert func.kind == "function"
    
    cls = parser.find_symbol("MyClass")
    assert cls is not None, "MyClass not found"
    assert cls.kind == "class"
    assert "method_a" in cls.children
    
    print("✓ test_python_parser passed")


def test_cpp_parser():
    """Test C/C++ parser."""
    code = '''
#include <stdio.h>
#include "mylib.h"

static int add(int a, int b) {
    return a + b;
}

class Calculator {
public:
    int multiply(int x, int y) {
        return x * y;
    }
};
'''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
        f.write(code)
        f.flush()
        parser = CppParser(f.name)
        parser.parse()
        os.unlink(f.name)
    
    assert len(parser.symbols) >= 2, f"Expected 2+ symbols, got {len(parser.symbols)}"
    assert len(parser.imports) >= 2
    assert "add" in [s.name for s in parser.symbols]
    
    print("✓ test_cpp_parser passed")


def test_tool_registry():
    """Test tool registry and validation."""
    registry = ToolRegistry()
    
    # Test schema generation
    tools = registry.to_openai_tools()
    assert len(tools) > 5, f"Expected 5+ tools, got {len(tools)}"
    
    # Test validation
    valid, error = registry.validate_args("read_file", {"path": "/test.py"})
    assert valid, f"Validation failed: {error}"
    
    valid, error = registry.validate_args("read_file", {})
    assert not valid, "Expected missing required param"
    
    print("✓ test_tool_registry passed")


def test_tool_call_parser():
    """Test tool call parsing."""
    xml_text = '''
<tool_call>
<name>read_file</name>
<params>{"path": "/test.py", "start_line": 10}</params>
</tool_call>
'''
    calls = ToolCallParser.parse(xml_text)
    assert len(calls) == 1
    assert calls[0]["name"] == "read_file"
    assert calls[0]["params"]["path"] == "/test.py"
    
    # JSON format
    json_text = '{"name": "run_tests", "params": {"test_path": "tests/"}}'
    calls = ToolCallParser.parse(json_text)
    assert len(calls) == 1
    assert calls[0]["name"] == "run_tests"
    
    print("✓ test_tool_call_parser passed")


def test_static_analyzer():
    """Test static code analyzer."""
    analyzer = StaticAnalyzer()
    
    code_with_secret = '''
API_KEY = "sk-abcdefghijklmnopqrstuvwxyz123456"
'''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code_with_secret)
        f.flush()
        report = analyzer.analyze_file(f.name)
        os.unlink(f.name)
    
    assert report.total_warnings > 0, "Expected secret detection warning"
    
    print("✓ test_static_analyzer passed")


def test_debugger_agent():
    """Test debugger agent."""
    debugger = DebuggerAgent()
    
    # Test build error analysis
    build_output = "src/main.c:42:15: error: undefined reference to 'foo_func'"
    result = debugger.analyze_build_error(build_output)
    assert result.status == "build_fail"
    assert result.error_line == 42
    assert len(result.suggestions) > 0
    
    # Test test failure analysis
    test_output = '''
Traceback (most recent call last):
  File "/app/tests/test_calc.py", line 15, in test_add
    assert add(1, 2) == 4
AssertionError: 3 != 4
'''
    result = debugger.analyze_test_failure(test_output)
    assert result.status == "test_fail"
    assert result.error_line == 15
    
    print("✓ test_debugger_agent passed")


def test_repository_analyzer():
    """Test repository analysis on the tinyllm source itself."""
    tinyllm_root = Path(__file__).resolve().parent.parent.parent / "src"
    if not tinyllm_root.exists():
        print("⊘ test_repository_analyzer skipped (src/ not found)")
        return
    
    analyzer = RepositoryAnalyzer(str(tinyllm_root))
    analyzer.scan()
    
    assert analyzer.metrics["total_files"] > 0
    assert analyzer.metrics["total_functions"] > 0
    
    print(f"✓ test_repository_analyzer passed ({analyzer.metrics['total_files']} files, "
          f"{analyzer.metrics['total_functions']} functions)")


if __name__ == "__main__":
    test_python_parser()
    test_cpp_parser()
    test_tool_registry()
    test_tool_call_parser()
    test_static_analyzer()
    test_debugger_agent()
    test_repository_analyzer()
    print("\n✅ All Phase 6 tests passed!")
