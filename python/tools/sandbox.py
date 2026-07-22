"""
sandbox.py — Secure sandboxed execution for tinyllm tools.

Supports Docker/Podman containerized execution for safety.
Falls back to subprocess with resource limits if containers unavailable.

Usage:
  python sandbox.py run --lang python --code "print(1+1)"
  python sandbox.py test --lang c --source path/to/code.c --tests path/to/tests.c
"""

import argparse
import json
import os
import resource
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional


class SandboxExecutor:
    """Secure code execution environment."""

    MEMORY_LIMIT_MB = 512
    TIME_LIMIT_SEC = 30
    MAX_OUTPUT_BYTES = 1024 * 1024  # 1 MB

    def __init__(self, use_container: bool = True):
        self.use_container = use_container
        self.container_runtime = self._detect_container_runtime()

    def _detect_container_runtime(self) -> Optional[str]:
        """Detect available container runtime (docker, podman, or none)."""
        for runtime in ['docker', 'podman']:
            if shutil.which(runtime):
                print(f"[sandbox] Using container runtime: {runtime}")
                return runtime
        print("[sandbox] No container runtime found. Using subprocess with limits.")
        return None

    def run_python(self, code: str, timeout: Optional[float] = None) -> dict:
        """Execute Python code in sandbox."""
        timeout = timeout or self.TIME_LIMIT_SEC

        if self.use_container and self.container_runtime:
            return self._run_container('python', code, timeout)
        else:
            return self._run_subprocess('python3', code, timeout)

    def run_c(self, code: str, timeout: Optional[float] = None) -> dict:
        """Compile and execute C code in sandbox."""
        timeout = timeout or self.TIME_LIMIT_SEC
        workdir = tempfile.mkdtemp(prefix="sandbox_c_")

        try:
            # Write source
            src = os.path.join(workdir, "prog.c")
            with open(src, 'w') as f:
                f.write(code)

            # Compile
            compile_result = subprocess.run(
                ['gcc', '-O2', '-o', os.path.join(workdir, 'prog'), src, '-lm'],
                capture_output=True, text=True, timeout=10,
            )

            if compile_result.returncode != 0:
                return {
                    'success': False,
                    'error': 'compilation_failed',
                    'stderr': compile_result.stderr[-2000:],
                    'stdout': '',
                    'time_ms': 0,
                }

            # Execute
            exec_result = self._run_binary(
                os.path.join(workdir, 'prog'), timeout
            )
            exec_result['compilation'] = 'ok'
            return exec_result

        finally:
            try: shutil.rmtree(workdir)
            except: pass

    def run_tests(self, code: str, test_code: str, language: str = 'python') -> dict:
        """Run code against test cases."""
        if language == 'python':
            full_code = f"{code}\n\n{test_code}\n\n"
            return self.run_python(full_code)
        elif language in ('c', 'cpp'):
            full_code = f"{code}\n\n{test_code}"
            return self.run_c(full_code)
        else:
            return {'success': False, 'error': f'unsupported language: {language}'}

    def _run_container(self, lang: str, code: str, timeout: float) -> dict:
        """Run code inside a container."""
        runtime = self.container_runtime
        if not runtime:
            return self._run_subprocess(lang, code, timeout)

        cmd = [
            runtime, 'run', '--rm',
            '--network', 'none',
            '--memory', f'{self.MEMORY_LIMIT_MB}m',
            '--cpus', '1',
            '--read-only',
            '--tmpfs', '/tmp:rw,noexec,nosuid,size=256M',
            '--security-opt', 'no-new-privileges',
            'alpine:latest',
            lang, '-c', code,
        ]

        try:
            t0 = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True, text=True,
                timeout=timeout + 5,  # extra for container startup
            )
            elapsed = (time.time() - t0) * 1000

            return {
                'success': result.returncode == 0,
                'stdout': result.stdout[-self.MAX_OUTPUT_BYTES:],
                'stderr': result.stderr[-self.MAX_OUTPUT_BYTES:],
                'returncode': result.returncode,
                'time_ms': elapsed,
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'timeout',
                'stdout': '',
                'stderr': f'Execution exceeded {timeout}s limit',
                'time_ms': timeout * 1000,
            }

    def _run_subprocess(self, lang: str, code: str, timeout: float) -> dict:
        """Run code via subprocess with resource limits."""
        def set_limits():
            """Set resource limits for child process."""
            try:
                limit_bytes = self.MEMORY_LIMIT_MB * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
                resource.setrlimit(resource.RLIMIT_CPU, (int(timeout) + 1, int(timeout) + 1))
            except Exception:
                pass

        try:
            t0 = time.time()
            result = subprocess.run(
                [lang, '-c', code],
                capture_output=True, text=True,
                timeout=timeout,
                preexec_fn=set_limits if os.name != 'nt' else None,
            )
            elapsed = (time.time() - t0) * 1000

            return {
                'success': result.returncode == 0,
                'stdout': result.stdout[-self.MAX_OUTPUT_BYTES:],
                'stderr': result.stderr[-self.MAX_OUTPUT_BYTES:],
                'returncode': result.returncode,
                'time_ms': elapsed,
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'timeout',
                'stdout': '',
                'stderr': f'Execution exceeded {timeout}s limit',
                'time_ms': timeout * 1000,
            }

    def _run_binary(self, binary_path: str, timeout: float) -> dict:
        """Run a compiled binary with limits."""
        try:
            t0 = time.time()
            result = subprocess.run(
                [binary_path],
                capture_output=True, text=True,
                timeout=timeout,
            )
            elapsed = (time.time() - t0) * 1000

            return {
                'success': result.returncode == 0,
                'stdout': result.stdout[-self.MAX_OUTPUT_BYTES:],
                'stderr': result.stderr[-self.MAX_OUTPUT_BYTES:],
                'returncode': result.returncode,
                'time_ms': elapsed,
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'timeout',
                'stdout': '',
                'stderr': f'Execution exceeded {timeout}s limit',
                'time_ms': timeout * 1000,
            }


def main():
    parser = argparse.ArgumentParser(description="Sandbox Executor for tinyllm")
    sub = parser.add_subparsers(dest='command')

    run_p = sub.add_parser('run')
    run_p.add_argument('--lang', default='python')
    run_p.add_argument('--code', required=True)
    run_p.add_argument('--timeout', type=float, default=30)
    run_p.add_argument('--no-container', action='store_true')

    test_p = sub.add_parser('test')
    test_p.add_argument('--lang', default='python')
    test_p.add_argument('--code', required=True)
    test_p.add_argument('--tests', required=True)
    test_p.add_argument('--timeout', type=float, default=30)

    args = parser.parse_args()
    executor = SandboxExecutor(use_container=not getattr(args, 'no_container', False))

    if args.command == 'run':
        if args.lang == 'python':
            result = executor.run_python(args.code, args.timeout)
        elif args.lang in ('c', 'cpp'):
            result = executor.run_c(args.code, args.timeout)
        else:
            result = {'success': False, 'error': f'unsupported language: {args.lang}'}
        print(json.dumps(result, indent=2))

    elif args.command == 'test':
        with open(args.tests) as f:
            test_code = f.read()
        result = executor.run_tests(args.code, test_code, args.lang)
        print(json.dumps(result, indent=2))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
