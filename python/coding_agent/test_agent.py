import tempfile
from pathlib import Path
from .workspace import Workspace

def test_workspace_blocks_escape():
    with tempfile.TemporaryDirectory() as d:
        w=Workspace(d)
        try:w.read('../secret')
        except PermissionError:return
        assert False

def test_workspace_write_and_read():
    with tempfile.TemporaryDirectory() as d:
        w=Workspace(d); w.write('a.txt','hello'); assert w.read('a.txt')=='hello'
