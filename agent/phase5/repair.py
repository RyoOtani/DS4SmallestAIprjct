
import json

class RepairLoop:
    def __init__(self, engineer, max_attempts=3):
        self.engineer=engineer
        self.max_attempts=max_attempts

    def diagnose(self, request, verification):
        context=json.dumps(verification, ensure_ascii=False)
        return self.engineer.ask("debugger", request, context)

    def run(self, request, test_command, execute_fix):
        history=[]
        for attempt in range(1,self.max_attempts+1):
            verification=self.engineer.verify(test_command)
            history.append({"attempt":attempt,"verification":verification})
            if verification.get("returncode")==0:
                return {"status":"passed","attempts":attempt,"history":history}
            diagnosis=self.diagnose(request, verification)
            history[-1]["diagnosis"]=diagnosis
            fixed=execute_fix(diagnosis, verification, attempt)
            history[-1]["fix"]=fixed
            if not fixed.get("ok",False):
                break
        return {"status":"failed","attempts":len(history),"history":history}
