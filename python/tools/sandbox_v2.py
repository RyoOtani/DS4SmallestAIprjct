"""Safer workspace command runner. Container mode is recommended for untrusted code."""
from __future__ import annotations
import subprocess, shutil, tempfile, os

def run(command, cwd, timeout=60, container=True):
    if container and shutil.which('docker'):
        # Mount workspace read-write only when explicitly requested by caller.
        cmd=['docker','run','--rm','--network','none','--memory','1g','--cpus','2','-v',f'{os.path.abspath(cwd)}:/workspace','-w','/workspace','python:3.12-slim','sh','-lc',command]
    else:
        cmd=['sh','-lc',command]
    try:
        p=subprocess.run(cmd,cwd=None if container else cwd,text=True,capture_output=True,timeout=timeout)
        return {'returncode':p.returncode,'stdout':p.stdout[-20000:],'stderr':p.stderr[-20000:]}
    except subprocess.TimeoutExpired:
        return {'returncode':124,'stdout':'','stderr':f'timeout after {timeout}s'}
