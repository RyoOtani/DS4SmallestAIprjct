
import json
from .provider import MockProvider
from .tools import WorkspaceTools
from .memory import AgentMemory

SYSTEM = """You are a careful AI software engineer.
Use repository evidence before proposing changes.
Prefer minimal diffs.
Never claim a change was made unless a tool actually made it.
Return concise, actionable plans and review findings."""

class AIEngineer:
    def __init__(self, workspace, provider=None, memory_path=None):
        self.tools=WorkspaceTools(workspace)
        self.provider=provider or MockProvider()
        self.memory=AgentMemory(memory_path or (self.tools.root/".tinyllm/memory.json"))

    def ask(self, role, request, context=""):
        messages=[
            {"role":"system","content":SYSTEM},
            {"role":"system","content":f"Role: {role}"},
            {"role":"user","content":f"Request:\n{request}\n\nRepository context:\n{context}"},
        ]
        return self.provider.complete(messages)

    def inspect(self, request):
        files=self.tools.search("", "*")[:100]
        context="\n".join(files)
        answer=self.ask("repository architect", request, context)
        self.memory.add("inspection", answer, {"request":request})
        return {"files":files,"analysis":answer}

    def verify(self, command="python -m pytest -q"):
        result=self.tools.run(command)
        self.memory.add("verification", json.dumps(result), {"command":command})
        return result

    def review(self, request, diff=None, verification=None):
        context=f"DIFF:\n{diff or self.tools.git_diff()}\nVERIFICATION:\n{verification or {}}"
        answer=self.ask("critical code reviewer", request, context)
        self.memory.add("review", answer, {"request":request})
        return answer

    def run_cycle(self, request, test_command="python -m pytest -q"):
        inspection=self.inspect(request)
        verification=self.verify(test_command)
        review=self.review(request, verification=verification)
        return {"inspection":inspection,"verification":verification,"review":review}
