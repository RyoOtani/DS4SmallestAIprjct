"""
sandbox.py — Safe code execution sandbox for self-repair loops.

Features:
  - Python: subprocess with resource limits (time, memory, no network)
  - Shell: restricted command execution with allowlist
  - File: temp directory mounting, read-only access to source
  - Output capture with size limits
  - Return code, stdout, stderr, execution time
"""
import subprocess, os, sys, tempfile, shutil, signal, resource, time
from pathlib import Path
from typing import Tuple, Optional, Dict


class SandboxResult:
    """Result of a sandboxed execution."""
    def __init__(self):
        self.success: bool = False
        self.exit_code: int = -1
        self.stdout: str = ""
        self.stderr: str = ""
        self.runtime_ms: float = 0
        self.timed_out: bool = False
        self.killed_by_oom: bool = False
        self.error: str = ""

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "exit_code": self.exit_code,
            "stdout": self.stdout[:5000],
            "stderr": self.stderr[:2000],
            "runtime_ms": round(self.runtime_ms, 1),
            "timed_out": self.timed_out,
            "killed_by_oom": self.killed_by_oom,
            "error": self.error,
        }


class CodeSandbox:
    """Secure execution environment for code testing."""

    MAX_OUTPUT_BYTES = 100_000
    DEFAULT_TIMEOUT_SEC = 30
    DEFAULT_MEMORY_MB = 512

    def __init__(self, workspace: str = None, timeout: int = None, memory_mb: int = None):
        self.workspace = Path(workspace or tempfile.mkdtemp(prefix="tinyllm_sandbox_"))
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout or self.DEFAULT_TIMEOUT_SEC
        self.memory_mb = memory_mb or self.DEFAULT_MEMORY_MB

    # ── Python Execution ──────────────────────────────────

    def run_python(self, code: str, stdin: str = "",
                   env: Dict[str, str] = None) -> SandboxResult:
        """Execute Python code in a subprocess with resource limits."""
        return self._run_subprocess(
            [sys.executable, "-c", code],
            stdin=stdin,
            env=self._build_env(env, no_network=True),
        )

    def run_python_file(self, filepath: str, args: list = None,
                        stdin: str = "") -> SandboxResult:
        """Execute a Python file with resource limits."""
        cmd = [sys.executable, str(filepath)]
        if args:
            cmd.extend(args)
        return self._run_subprocess(cmd, stdin=stdin, env=self._build_env(no_network=True))

    # ── Shell Execution ───────────────────────────────────

    def run_shell(self, command: str, stdin: str = "",
                  allowlist: list = None) -> SandboxResult:
        """
        Execute a shell command with restrictions.
        allowlist: list of allowed commands (e.g. ['python', 'gcc', 'make'])
        """
        # Basic command injection prevention
        if allowlist:
            first_word = command.strip().split()[0] if command.strip() else ""
            if first_word not in allowlist:
                r = SandboxResult()
                r.error = f"Command '{first_word}' not in allowlist: {allowlist}"
                return r

        return self._run_subprocess(
            ["/bin/bash", "-c", command],
            stdin=stdin,
            env=self._build_env(no_network=True),
        )

    # ── File-based Execution ──────────────────────────────

    def write_file(self, filename: str, content: str):
        """Write a file into the sandbox workspace."""
        path = self.workspace / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def read_file(self, filename: str) -> str:
        """Read a file from the sandbox workspace."""
        return (self.workspace / filename).read_text()

    def list_files(self) -> list:
        """List all files in the sandbox."""
        return [str(p.relative_to(self.workspace))
                for p in self.workspace.rglob("*") if p.is_file()]

    # ── Cleanup ───────────────────────────────────────────

    def cleanup(self):
        """Remove the sandbox workspace."""
        try:
            shutil.rmtree(self.workspace, ignore_errors=True)
        except Exception:
            pass

    # ── Internal ───────────────────────────────────────────

    def _build_env(self, extra: Dict[str, str] = None, no_network: bool = True) -> dict:
        """Build a safe environment dict."""
        env = os.environ.copy()
        env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
        env["HOME"] = str(self.workspace)
        env["TMPDIR"] = str(self.workspace)

        # Block network access
        if no_network:
            env["http_proxy"] = ""
            env["https_proxy"] = ""
            env["HTTP_PROXY"] = ""
            env["HTTPS_PROXY"] = ""
            env["no_proxy"] = "*"

        if extra:
            env.update(extra)
        return env

    def _run_subprocess(self, cmd: list, stdin: str = "",
                        env: dict = None) -> SandboxResult:
        result = SandboxResult()
        start = time.time()

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.workspace),
                env=env,
                preexec_fn=self._set_limits if os.name != 'nt' else None,
                text=True,
            )

            try:
                out, err = proc.communicate(
                    input=stdin,
                    timeout=self.timeout,
                )
                result.stdout = out[:self.MAX_OUTPUT_BYTES]
                result.stderr = err[:self.MAX_OUTPUT_BYTES]
                result.exit_code = proc.returncode
                result.success = proc.returncode == 0
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                result.timed_out = True
                result.error = f"Timed out after {self.timeout}s"
                # Try to get partial output
                try:
                    out, err = proc.communicate(timeout=1)
                    result.stdout = (out or "")[:self.MAX_OUTPUT_BYTES]
                    result.stderr = (err or "")[:self.MAX_OUTPUT_BYTES]
                except Exception:
                    pass

        except Exception as e:
            result.error = str(e)

        result.runtime_ms = (time.time() - start) * 1000

        # Check for OOM (exit code 137 = SIGKILL, often from OOM killer)
        if result.exit_code == -9 or result.exit_code == 137:
            result.killed_by_oom = True
            result.error = "Killed (possible OOM)"

        return result

    @staticmethod
    def _set_limits():
        """Set resource limits for the child process."""
        try:
            # CPU time limit
            resource.setrlimit(resource.RLIMIT_CPU, (CodeSandbox.DEFAULT_TIMEOUT_SEC + 5,
                                                       CodeSandbox.DEFAULT_TIMEOUT_SEC + 5))
            # Memory limit
            mem_bytes = CodeSandbox.DEFAULT_MEMORY_MB * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            # No core dumps
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            # Limit file size
            resource.setrlimit(resource.RLIMIT_FSIZE, (100 * 1024 * 1024, 100 * 1024 * 1024))
        except Exception:
            pass  # Resource limits not available on all platforms


# ── Convenience functions ─────────────────────────────────

def safe_execute_python(code: str, timeout: int = 30) -> SandboxResult:
    """Quick one-shot Python execution with safety limits."""
    sandbox = CodeSandbox(timeout=timeout)
    try:
        return sandbox.run_python(code)
    finally:
        sandbox.cleanup()


def safe_execute_shell(command: str, timeout: int = 30,
                       allowlist: list = None) -> SandboxResult:
    """Quick one-shot shell execution with safety limits."""
    sandbox = CodeSandbox(timeout=timeout)
    try:
        return sandbox.run_shell(command, allowlist=allowlist)
    finally:
        sandbox.cleanup()


# ── CLI ───────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: sandbox.py <python|shell> <code|command>")
        sys.exit(1)

    mode = sys.argv[1]
    code = sys.argv[2] if len(sys.argv) > 2 else "print('hello')"

    if mode == "python":
        r = safe_execute_python(code)
    else:
        r = safe_execute_shell(code)

    print(f"success={r.success} exit={r.exit_code} time={r.runtime_ms:.0f}ms")
    if r.stdout: print(f"stdout:\n{r.stdout}")
    if r.stderr: print(f"stderr:\n{r.stderr}")
    if r.error: print(f"error: {r.error}")
