
from agent.phase4.engine import AIEngineer
from agent.phase4.provider import MockProvider

def test_phase4_inspect_and_verify(tmp_path):
    (tmp_path/"hello.py").write_text("print('hi')\n", encoding="utf-8")
    e=AIEngineer(tmp_path, provider=MockProvider("analysis"))
    result=e.run_cycle("Inspect this repository", test_command="python -c \"print('ok')\"")
    assert result["verification"]["returncode"] == 0
    assert "hello.py" in result["inspection"]["files"]
    assert result["review"] == "analysis"

def test_workspace_escape_is_blocked(tmp_path):
    e=AIEngineer(tmp_path, provider=MockProvider())
    try:
        e.tools.read_file("../outside")
    except ValueError:
        return
    assert False
