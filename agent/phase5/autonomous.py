
import json
from pathlib import Path
from .patching import apply_unified_patch, PatchError
from .repair import RepairLoop
from ..phase4.engine import AIEngineer
from ..phase4.provider import MockProvider

class AutonomousCodingAgent:
    """
    Phase 5 end-to-end loop.
    The model is asked for structured actions; edits are applied only after validation.
    """

    def __init__(self, workspace, provider=None, max_attempts=3):
        self.engineer=AIEngineer(workspace, provider=provider or MockProvider())
        self.root=Path(workspace).resolve()
        self.max_attempts=max_attempts

    def checkpoint(self):
        result=self.engineer.tools.run("git status --porcelain")
        return {"ok":result["returncode"]==0,"status":result["stdout"]}

    def apply_patch(self, patch_text):
        try:
            files=apply_unified_patch(self.root, patch_text)
            return {"ok":True,"files":files}
        except Exception as e:
            return {"ok":False,"error":str(e)}

    def plan(self, request):
        inspection=self.engineer.inspect(request)
        return self.engineer.ask(
            "planner and architect",
            request,
            "Repository files:\n"+json.dumps(inspection["files"][:200])+
            "\nInspection:\n"+inspection["analysis"]
        )

    def run(self, request, test_command="python -m pytest -q", patch_provider=None):
        checkpoint=self.checkpoint()
        plan=self.plan(request)
        history=[{"stage":"plan","plan":plan}]
        provider=patch_provider or self.engineer.provider

        # A provider can return a unified diff directly. This keeps the edit boundary explicit.
        edit_prompt=("Return ONLY a unified diff patch for the requested change. "
                     "Do not include markdown fences.\nRequest: "+request+
                     "\nPlan:\n"+plan)
        patch=provider.complete([
            {"role":"system","content":"You are a precise code editor. Output a valid unified diff only."},
            {"role":"user","content":edit_prompt}
        ])
        applied=self.apply_patch(patch)
        history.append({"stage":"apply_patch","result":applied})
        if not applied["ok"]:
            return {"status":"patch_failed","checkpoint":checkpoint,"history":history}

        repair=RepairLoop(self.engineer,self.max_attempts)
        def execute_fix(diagnosis, verification, attempt):
            fix_prompt=("Produce ONLY a unified diff to fix the failing tests. "
                        "Do not include markdown fences.\nRequest: "+request+
                        "\nDiagnosis:\n"+diagnosis+
                        "\nVerification:\n"+json.dumps(verification,ensure_ascii=False))
            diff=provider.complete([
                {"role":"system","content":"You are a bug-fixing code editor. Output a valid unified diff only."},
                {"role":"user","content":fix_prompt}
            ])
            result=self.apply_patch(diff)
            return result
        repair_result=repair.run(request,test_command,execute_fix)
        history.append({"stage":"repair_loop","result":repair_result})

        verification=self.engineer.verify(test_command)
        review=self.engineer.review(request,verification=verification)
        history.append({"stage":"final_review","verification":verification,"review":review})

        status="completed" if verification.get("returncode")==0 else "failed"
        return {"status":status,"checkpoint":checkpoint,"history":history}
