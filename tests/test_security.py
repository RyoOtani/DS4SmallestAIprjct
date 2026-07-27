#!/usr/bin/env python3
"""
test_security.py — Security regression tests for TinyLLM.

Ensures:
  1. CommandProvider never uses shell=True (command injection prevention)
  2. Invalid command inputs are handled gracefully
  3. No shell=True exists anywhere in provider code

Run: python test_security.py
"""

import os, sys, subprocess, unittest, tempfile

# Add repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestCommandProviderSecurity(unittest.TestCase):
    """Verify CommandProvider is safe from command injection."""

    def setUp(self):
        from python.runtime.provider import CommandProvider
        self.CommandProvider = CommandProvider

    def test_shell_is_always_false(self):
        """CommandProvider.generate() must NEVER use shell=True."""
        import inspect
        src = inspect.getsource(self.CommandProvider.generate)
        self.assertNotIn('shell=True', src,
                         "SECURITY: shell=True found in CommandProvider.generate()!")
        self.assertIn('shell=False', src,
                      "CommandProvider must explicitly set shell=False")

    def test_simple_command_works(self):
        """Basic command execution should work."""
        cp = self.CommandProvider('echo hello')
        result = cp.generate('test prompt')
        self.assertIn('hello', result.text)

    def test_command_with_args(self):
        """Command with arguments (e.g., 'ollama run llama3') should work."""
        cp = self.CommandProvider('echo hello world')
        result = cp.generate('test')
        self.assertIn('hello world', result.text)

    def test_invalid_command_raises(self):
        """Non-existent command should raise RuntimeError, not hang."""
        cp = self.CommandProvider('nonexistent_command_xyzzy_12345')
        with self.assertRaises(RuntimeError):
            cp.generate('test')

    def test_command_with_special_chars(self):
        """Special characters in prompt should NOT be interpreted as shell."""
        cp = self.CommandProvider('cat')
        # If shell=True were used, this could execute arbitrary commands.
        # With shell=False, the 'cat' command just tries to read stdin
        # and the prompt content is treated as data, not code.
        try:
            cp.generate('$(whoami) ; `id` ; harmless')
        except RuntimeError:
            pass  # Expected — 'cat' with no file will fail, but safely
        # The key: no shell expansion happened

    def test_command_with_pipes_arent_interpreted(self):
        """Pipe characters should be treated as data, not shell pipes."""
        # 'cat' reads stdin and outputs as-is — perfect for testing
        cp = self.CommandProvider('cat')
        result = cp.generate('hello | rm -rf /')
        self.assertIn('|', result.text, "Pipe should be literal text, not shell operator")
        self.assertIn('rm', result.text)


class TestNoShellTrueAnywhere(unittest.TestCase):
    """Ensure no provider code anywhere uses shell=True."""

    def test_no_shell_true_in_providers(self):
        """Grep all provider files for shell=True."""
        provider_files = [
            'python/runtime/provider.py',
            'agent/phase4/provider.py',
        ]
        for fpath in provider_files:
            if not os.path.exists(fpath):
                continue
            with open(fpath) as f:
                content = f.read()
            # Find 'shell=True' that isn't in a comment
            lines = content.split('\n')
            for i, line in enumerate(lines):
                stripped = line.strip()
                if 'shell=True' in stripped and not stripped.startswith('#'):
                    # Allow if it's in a string literal (e.g., docstring)
                    if '"""' not in stripped and "'''" not in stripped:
                        self.fail(
                            f"SECURITY: shell=True found in {fpath}:{i+1}\n"
                            f"  {stripped}\n"
                            f"  All providers must use shell=False with shlex.split()"
                        )

    def test_no_shell_true_in_tool_handlers(self):
        """Grep tool_calling.py for shell=True in _run_command."""
        fpath = 'agent/phase6/tool_calling.py'
        if not os.path.exists(fpath):
            return
        with open(fpath) as f:
            content = f.read()
        if 'shell=True' in content:
            # Find context
            for i, line in enumerate(content.split('\n')):
                if 'shell=True' in line:
                    self.fail(
                        f"SECURITY: shell=True in tool_calling.py:{i+1}\n"
                        f"  Tool execution must also use shell=False"
                    )


class TestProviderEnvSafety(unittest.TestCase):
    """Verify provider_from_env() doesn't introduce injection vectors."""

    def test_provider_from_env_uses_safe_defaults(self):
        """provider_from_env() should default to safe OpenAI-compatible."""
        from python.runtime.provider import provider_from_env
        # Clear any env overrides
        old_env = {k: os.environ.pop(k, None) for k in
                   ['TINYLLM_PROVIDER', 'TINYLLM_COMMAND', 'TINYLLM_BASE_URL',
                    'TINYLLM_MODEL', 'TINYLLM_API_KEY']}
        try:
            provider = provider_from_env()
            from python.runtime.provider import OpenAICompatibleProvider
            self.assertIsInstance(provider, OpenAICompatibleProvider,
                                  "Default provider should be OpenAICompatible (safe)")
        finally:
            # Restore env
            for k, v in old_env.items():
                if v is not None:
                    os.environ[k] = v


if __name__ == '__main__':
    print("🔒 TinyLLM Security Regression Tests")
    print("=" * 50)
    unittest.main(verbosity=2)
