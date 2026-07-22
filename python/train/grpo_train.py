"""
grpo_train.py — GRPO (Group Relative Policy Optimization) for code generation.

GRPO reinforces correct code generation by:
  1. Generating N candidate solutions for a coding problem
  2. Evaluating each via test execution (reward = pass rate)
  3. Optimizing the policy to favor high-reward generations

Unlike PPO, GRPO doesn't need a separate value network —
it uses group-relative advantages within each batch.

Mathematically:
  advantage_i = (reward_i - mean(rewards)) / std(rewards)
  loss = -E[advantage_i * log_prob(action_i)]

Usage:
  python grpo_train.py --model student.gguf --problems problems.jsonl \
                       --output finetuned_model/
"""

import argparse
import json
import math
import os
import random
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class GRPOConfig:
    """GRPO training configuration."""
    model_path: str = "output/student_model"
    problems_path: str = "data/code_problems.jsonl"
    output_path: str = "output/grpo_finetuned"

    # Generation
    n_candidates: int = 8              # N candidates per problem (group size)
    max_new_tokens: int = 1024
    temperature: float = 0.8
    top_p: float = 0.95

    # Training
    learning_rate: float = 1e-6
    clip_epsilon: float = 0.2         # PPO-style clipping
    kl_beta: float = 0.01             # KL divergence penalty
    max_epochs: int = 3
    batch_size: int = 4               # problems per batch (each has N candidates)

    # Reward
    reward_timeout: float = 30.0       # seconds per test execution
    reward_pass_bonus: float = 1.0     # bonus for all tests passing
    reward_compile_bonus: float = 0.3  # bonus for successful compilation

    # Self-correction
    self_correct: bool = True
    max_corrections: int = 3

    # LoRA
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: float = 32.0

    # Mixed precision
    use_bfloat16: bool = True


class CodeExecutionEnv:
    """Sandboxed code execution for reward computation."""

    def __init__(self, timeout: float = 30.0, workdir: Optional[str] = None):
        self.timeout = timeout
        self.workdir = workdir or tempfile.mkdtemp(prefix="grpo_exec_")

    def execute_python(self, code: str, test_code: str) -> dict:
        """Execute Python code with tests. Returns {pass: bool, output: str, errors: str}."""
        full_code = f"{code}\n\n{test_code}\n\n"

        # Write to temp file
        tmpfile = os.path.join(self.workdir, f"test_{random.randint(0,10**9)}.py")
        with open(tmpfile, 'w') as f:
            f.write(full_code)

        try:
            result = subprocess.run(
                ['python', tmpfile],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.workdir,
            )
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {'success': False, 'stdout': '', 'stderr': 'TIMEOUT', 'returncode': -1}
        finally:
            try:
                os.remove(tmpfile)
            except OSError:
                pass

    def execute_c(self, code: str, test_code: str) -> dict:
        """Compile and execute C code with tests."""
        c_file = os.path.join(self.workdir, "prog.c")
        bin_file = os.path.join(self.workdir, "prog")

        # Write C source
        full_source = f"{code}\n\n{test_code}"
        with open(c_file, 'w') as f:
            f.write(full_source)

        # Compile
        compile_result = subprocess.run(
            ['gcc', '-o', bin_file, c_file, '-lm', '-Wall'],
            capture_output=True, text=True, timeout=10.0,
        )

        if compile_result.returncode != 0:
            return {'success': False, 'compile_error': True,
                    'stderr': compile_result.stderr, 'stdout': ''}

        # Execute
        try:
            run_result = subprocess.run(
                [bin_file],
                capture_output=True, text=True, timeout=self.timeout,
            )
            return {
                'success': run_result.returncode == 0,
                'compile_error': False,
                'stdout': run_result.stdout,
                'stderr': run_result.stderr,
                'returncode': run_result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {'success': False, 'stdout': '', 'stderr': 'TIMEOUT', 'returncode': -1}
        finally:
            try: os.remove(c_file)
            except: pass
            try: os.remove(bin_file)
            except: pass


class RewardCalculator:
    """Computes reward from execution results."""

    def __init__(self, config: GRPOConfig):
        self.config = config
        self.exec_env = CodeExecutionEnv(timeout=config.reward_timeout)

    def compute_reward(self, code: str, problem: dict) -> float:
        """Compute reward for a generated code solution.

        Reward components:
          - Each passing test: +1.0 / n_tests
          - All tests pass: +bonus
          - Successful compilation: +compile_bonus
          - Syntax error: -0.2 penalty
          - Timeout: -0.5 penalty
        """
        tests = problem.get('tests', [])
        language = problem.get('language', 'python')
        test_code = problem.get('test_code', '')

        if not tests and not test_code:
            return 0.0  # no tests to evaluate

        if language == 'python':
            result = self.exec_env.execute_python(code, test_code if test_code else self._format_tests(tests))
        elif language in ('c', 'cpp'):
            result = self.exec_env.execute_c(code, test_code if test_code else self._format_tests(tests))
        else:
            result = self.exec_env.execute_python(code, test_code)

        reward = 0.0

        # Compilation
        if result.get('compile_error'):
            reward -= 0.2
            return reward

        reward += self.config.reward_compile_bonus

        # Test results
        if result.get('success'):
            if tests:
                reward += self.config.reward_pass_bonus
            else:
                reward += 1.0
        elif result.get('returncode', 0) == -1:  # Timeout
            reward -= 0.5
        else:
            # Partial credit: count passing assertions
            output = result.get('stdout', '') + result.get('stderr', '')
            pass_count = output.count('PASS') if 'PASS' in output else 0
            fail_count = output.count('FAIL') if 'FAIL' in output else 0
            total_checks = pass_count + fail_count
            if total_checks > 0:
                reward += pass_count / total_checks * 0.5

        return reward

    @staticmethod
    def _format_tests(tests: list) -> str:
        """Format test cases into executable code."""
        lines = []
        for test in tests:
            inputs = test.get('input', '')
            expected = test.get('expected', '')
            func_name = test.get('function', 'solution')
            lines.append(f"result = {func_name}(*{inputs})" if isinstance(inputs, list) else f"result = {func_name}({inputs})")
            lines.append(f"assert result == {expected}, f'FAIL: got {{result}}, expected {{expected}}'")
            lines.append("print('PASS')")
        return '\n'.join(lines)


class GRPOTrainer:
    """GRPO training loop with self-correction."""

    def __init__(self, config: GRPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reward_calc = RewardCalculator(config)
        self._load_model()
        self._load_problems()

    def _load_model(self):
        """Load the student model for further GRPO fine-tuning."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.bfloat16 if self.config.use_bfloat16 else torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, trust_remote_code=True
        )

        if self.config.use_lora:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        self.ref_model = None  # reference model for KL penalty

    def _load_problems(self):
        """Load coding problems for training."""
        with open(self.config.problems_path) as f:
            self.problems = [json.loads(line) for line in f]
        print(f"Loaded {len(self.problems)} problems")

    def generate_candidates(self, problem: dict) -> list[str]:
        """Generate N candidate solutions for a problem."""
        prompt = self._build_prompt(problem)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        candidates = []
        for _ in range(self.config.n_candidates):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            code = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
            )
            candidates.append(code)

        return candidates

    def _build_prompt(self, problem: dict) -> str:
        """Build a code-generation prompt."""
        description = problem.get('description', problem.get('prompt', ''))
        language = problem.get('language', 'python')
        signature = problem.get('signature', '')

        return (
            f"Write {language} code to solve the following problem.\n\n"
            f"## Problem\n{description}\n\n"
            f"## Function Signature\n```{language}\n{signature}\n```\n\n"
            f"## Solution\n```{language}\n"
        )

    def compute_grpo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Compute GRPO loss with PPO-style clipping."""
        ratio = torch.exp(log_probs - old_log_probs)

        # Clipped objective
        eps = self.config.clip_epsilon
        clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
        loss = -torch.min(ratio * advantages, clipped * advantages).mean()

        # KL penalty
        if self.config.kl_beta > 0 and self.ref_model is not None:
            kl = (log_probs - old_log_probs).mean()
            loss += self.config.kl_beta * kl

        return loss

    def self_correct(self, code: str, test_output: str, problem: dict) -> str:
        """Self-correction: ask the model to fix its code based on test failures."""
        correction_prompt = (
            f"The following code failed tests:\n\n```\n{code}\n```\n\n"
            f"Test output:\n{test_output}\n\n"
            f"Please fix the code:\n\n```\n"
        )

        inputs = self.tokenizer(correction_prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        corrected = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )
        return corrected

    def train_step(self, problem: dict) -> dict[str, float]:
        """Single GRPO training step on one problem."""
        # 1. Generate N candidates
        candidates = self.generate_candidates(problem)

        # 2. Evaluate all candidates
        rewards = []
        for code in candidates:
            reward = self.reward_calc.compute_reward(code, problem)

            # Self-correction
            if self.config.self_correct and reward < 0.5:
                for _ in range(self.config.max_corrections):
                    test_output = self.reward_calc.exec_env.execute_python(code, "").get('stderr', '')
                    code = self.self_correct(code, test_output, problem)
                    new_reward = self.reward_calc.compute_reward(code, problem)
                    if new_reward > reward:
                        reward = new_reward
                        break

            rewards.append(reward)

        rewards_t = torch.tensor(rewards, device=self.device)
        mean_r = rewards_t.mean()
        std_r = rewards_t.std() + 1e-8
        advantages = (rewards_t - mean_r) / std_r

        # 3. Optimize — compute log_probs for each candidate
        total_loss = torch.tensor(0.0, device=self.device)
        n_better = (rewards_t > mean_r).sum().item()

        for i, (code, adv) in enumerate(zip(candidates, advantages)):
            prompt = self._build_prompt(problem) + code
            inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True,
                                    max_length=2048).to(self.device)

            outputs = self.model(**inputs, labels=inputs['input_ids'])
            log_prob = -outputs.loss  # negative NLL
            old_log_prob = log_prob.detach()

            loss = self.compute_grpo_loss(log_prob.unsqueeze(0), old_log_prob.unsqueeze(0),
                                          adv.unsqueeze(0))
            total_loss = total_loss + loss

        total_loss = total_loss / self.config.n_candidates
        total_loss.backward()

        return {
            'loss': total_loss.item(),
            'mean_reward': mean_r.item(),
            'max_reward': rewards_t.max().item(),
            'min_reward': rewards_t.min().item(),
            'better_than_mean': n_better,
        }

    def train(self):
        """Main training loop."""
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
        )

        for epoch in range(self.config.max_epochs):
            random.shuffle(self.problems)
            pbar = tqdm(self.problems[:100], desc=f"Epoch {epoch+1}")

            for i, problem in enumerate(pbar):
                if i % self.config.batch_size == 0:
                    optimizer.zero_grad()

                metrics = self.train_step(problem)
                pbar.set_postfix(metrics)

                if (i + 1) % self.config.batch_size == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                # Save periodically
                if i > 0 and i % 50 == 0:
                    self.model.save_pretrained(f"{self.config.output_path}_step{i}")

            self.model.save_pretrained(f"{self.config.output_path}_epoch{epoch+1}")

        self.model.save_pretrained(self.config.output_path)
        print(f"✓ GRPO training complete. Model saved to {self.config.output_path}")


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for tinyllm")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--problems', type=str, required=True)
    parser.add_argument('--output', type=str, default='output/grpo_model')
    parser.add_argument('--n-candidates', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--self-correct', action='store_true', default=True)
    parser.add_argument('--no-lora', dest='use_lora', action='store_false', default=True)

    args = parser.parse_args()

    config = GRPOConfig(
        model_path=args.model,
        problems_path=args.problems,
        output_path=args.output,
        n_candidates=args.n_candidates,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        temperature=args.temperature,
        self_correct=args.self_correct,
        use_lora=args.use_lora,
    )

    trainer = GRPOTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
