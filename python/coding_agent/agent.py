"""Plan/act/observe coding agent with bounded self-repair."""
from __future__ import annotations
import json, re
from .tools import Toolset
from ..runtime.provider import Provider

SYSTEM = """You are TinyLLM Coding Agent v2. You modify software in a workspace.
Be precise. Inspect before editing. Prefer small patches. After changes, run tests/build.
Return exactly one JSON object with either:
{"action":"tool","name":"TOOL","args":{...}}
or {"action":"final","message":"..."}.
Do not claim success without verification. Available tools:\n{tools}"""

class CodingAgent:
    def __init__(self, provider: Provider, tools: Toolset, max_steps=30):
        self.provider=provider; self.tools=tools; self.max_steps=max_steps

    def run(self, task: str):
        history=[]
        tools=json.dumps(self.tools.definitions(),ensure_ascii=False)
        prompt=f"Task: {task}\n\nStart by inspecting the repository."
        for step in range(self.max_steps):
            r=self.provider.generate(prompt, SYSTEM.format(tools=tools)).text.strip()
            obj=self._parse(r)
            if obj.get('action')=='final':
                return {'success':True,'steps':step+1,'message':obj.get('message',''),'history':history}
            if obj.get('action')!='tool':
                prompt='Invalid response. Output only the required JSON action object.'; continue
            try:
                result=self.tools.execute(obj['name'],obj.get('args',{}))
            except Exception as e:
                result={'error':type(e).__name__+': '+str(e)}
            history.append({'step':step+1,'tool':obj['name'],'args':obj.get('args',{}),'result':result})
            prompt=(f"Task: {task}\n\nPrevious action: {json.dumps(obj,ensure_ascii=False)}\n"
                    f"Tool result: {json.dumps(result,ensure_ascii=False)}\n"
                    "Decide the next action. If code was changed, verify it with tests/build before finalizing.")
        return {'success':False,'steps':self.max_steps,'message':'max steps reached','history':history}

    @staticmethod
    def _parse(text):
        text=re.sub(r'^```(?:json)?|```$','',text.strip(),flags=re.M).strip()
        m=re.search(r'\{.*\}',text,re.S)
        if not m: return {'action':'invalid'}
        try:return json.loads(m.group(0))
        except:return {'action':'invalid'}
