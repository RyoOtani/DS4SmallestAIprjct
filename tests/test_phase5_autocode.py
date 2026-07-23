
from agent.phase5.patching import apply_unified_patch
from agent.phase5.autonomous import AutonomousCodingAgent
from agent.phase4.provider import MockProvider

def test_unified_patch(tmp_path):
    f=tmp_path/"hello.py"
    f.write_text("a=1\nb=2\n",encoding="utf-8")
    patch="--- a/hello.py\n+++ b/hello.py\n@@ -1,2 +1,2 @@\n-a=1\n+a=10\n b=2\n"
    assert apply_unified_patch(tmp_path,patch)==["hello.py"]
    assert f.read_text()=="a=10\nb=2\n"

def test_autonomous_agent_rejects_invalid_patch(tmp_path):
    agent=AutonomousCodingAgent(tmp_path,MockProvider("not a patch"))
    result=agent.run("Change a file",test_command="python -c \"print('ok')\"")
    assert result["status"]=="patch_failed"

def test_autonomous_agent_applies_patch_and_verifies(tmp_path):
    f=tmp_path/"hello.py"
    f.write_text("a=1\n",encoding="utf-8")
    class Provider:
        def __init__(self): self.calls=0
        def complete(self,messages,**kwargs):
            self.calls+=1
            if self.calls==1: return "plan"
            return "--- a/hello.py\n+++ b/hello.py\n@@ -1,1 +1,1 @@\n-a=1\n+a=2\n"
    provider=Provider()
    agent=AutonomousCodingAgent(tmp_path,provider)
    result=agent.run("Change a to 2",test_command="python -c \"assert open('hello.py').read().strip() == 'a=2'\"")
    assert result["status"]=="completed"
    assert f.read_text()=="a=2\n"
