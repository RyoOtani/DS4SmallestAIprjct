"""Lightweight repository indexer: files, symbols, imports, and searchable chunks."""
from __future__ import annotations
import ast, json, re
from pathlib import Path

EXT={'.py','.c','.h','.cpp','.hpp','.js','.jsx','.ts','.tsx','.java','.go','.rs'}
SKIP={'.git','node_modules','__pycache__','build','dist'}

def iter_files(root):
    for p in Path(root).rglob('*'):
        if p.is_file() and p.suffix in EXT and not any(x in SKIP for x in p.parts): yield p

def symbols(text,suffix):
    if suffix=='.py':
        try:
            tree=ast.parse(text); out=[]
            for n in ast.walk(tree):
                if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                    out.append({'name':n.name,'kind':type(n).__name__,'line':n.lineno})
            return out
        except: pass
    pat=r'\b(?:class|struct|enum|function|def|fn)\s+([A-Za-z_][A-Za-z0-9_]*)|\b(?:int|void|char|float|double|bool)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\('
    return [{'name':a or b,'kind':'symbol','line':i} for i,l in enumerate(text.splitlines(),1) for a,b in re.findall(pat,l)]

def build(root,out='code_index/index.jsonl'):
    root=Path(root); Path(out).parent.mkdir(parents=True,exist_ok=True); n=0
    with open(out,'w',encoding='utf8') as f:
        for p in iter_files(root):
            try:text=p.read_text(errors='replace')
            except:continue
            rec={'path':str(p.relative_to(root)),'language':p.suffix,'symbols':symbols(text,p.suffix),'imports':re.findall(r'^(?:import|from|#include)\s+([^\n]+)',text,re.M),'text':text[:50000]}
            f.write(json.dumps(rec,ensure_ascii=False)+'\n'); n+=1
    print(f'Indexed {n} files -> {out}')

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('-o','--output',default='code_index/index.jsonl');a=p.parse_args();build(a.root,a.output)
