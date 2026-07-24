#!/usr/bin/env python3
"""
TinyLLM Distillation Dataset Generator

Generates a large-scale distillation dataset using a teacher model.
This script calls the DeepSeek-V4-Flash-Pro (or any OpenAI-compatible API)
to create high-quality training data for student model distillation.

Usage:
  python generate_large_dataset.py --output large_dataset.jsonl --n-samples 1000
  python generate_large_dataset.py --output large_dataset.jsonl --teacher deepseek --n-samples 50000
"""

import argparse
import json
import os
import random
import time
import sys
from pathlib import Path
from typing import Optional


# ── Seed prompts for data generation ────────────────────────────
CODE_TOPICS = [
    "Implement a function that", "Write a class that", "Create a program that",
    "Implement an algorithm to", "Write a utility that", "Develop a module for",
    "Create a data structure that", "Write a decorator that",
]

INSTRUCTION_TOPICS = [
    "Explain the concept of", "Compare and contrast",
    "Describe how", "What is the difference between",
    "How does", "Why is", "What are the advantages of",
]

LANGUAGES = ["Python", "C", "C++", "JavaScript", "TypeScript", "Rust", "Go", "Java"]

# ── Template library (extend these for more variety) ────────────
CODE_TEMPLATES = [
    "Implement {function} in {lang} that processes {input} and returns {output}.",
    "Write a {lang} class that manages {resource} with create, read, update, and delete operations.",
    "Create a {lang} function that validates {input_type} according to these rules: {rules}.",
    "Implement a {lang} program that reads {data_source}, transforms it using {transformation}, and outputs {result_format}.",
    "Write a {lang} module that provides {feature} with thread-safe access and proper error handling.",
    "Implement an algorithm in {lang} to find the {problem_goal} given {constraints}.",
    "Create a {lang} utility that monitors {target} and triggers {action} when {condition} occurs.",
    "Write a {lang} library that implements the {pattern} design pattern for {use_case}.",
]

INSTRUCTION_TEMPLATES = [
    "Explain {topic} with a concrete example. Include a diagram description and code snippet if applicable.",
    "Compare {concept_a} and {concept_b}. When should each be used? Provide trade-offs.",
    "Describe the architecture of {system} and explain how its components interact.",
    "What are the key considerations when designing {system_type}? Include performance, security, and maintainability.",
    "Explain how {technology} works under the hood. Include implementation details and common pitfalls.",
    "Walk through the process of {process} step by step. Include error handling at each step.",
    "Design a solution for {problem}. Consider scalability, reliability, and cost trade-offs.",
]

# ── Data generation ─────────────────────────────────────────────
FILL_VALUES = {
    'function': ['binary search', 'quick sort', 'memoized fibonacci', 'LRU cache',
                 'URL shortener', 'rate limiter', 'task scheduler', 'event emitter',
                 'dependency resolver', 'configuration parser', 'template engine',
                 'connection pool', 'circuit breaker', 'retry handler'],
    'input': ['a list of numbers', 'a text file', 'JSON data', 'CSV content',
              'user input', 'HTTP requests', 'database records', 'stream data'],
    'output': ['a sorted array', 'transformed data', 'validation results',
               'aggregated statistics', 'a formatted report', 'API response'],
    'resource': ['database connections', 'file handles', 'network sockets',
                 'user sessions', 'cache entries', 'task queues'],
    'input_type': ['email addresses', 'phone numbers', 'credit card numbers',
                   'IP addresses', 'URLs', 'JSON payloads', 'XML documents'],
    'rules': ['format validation, length check, checksum verification',
              'whitelist filtering, regex matching, type coercion',
              'schema validation, required fields check, uniqueness constraint'],
    'data_source': ['a CSV file', 'a JSON API', 'a SQL database', 'a Redis cache',
                    'a Kafka stream', 'a log file', 'stdin input'],
    'transformation': ['a mapping function', 'a filtering pipeline', 'aggregation',
                       'normalization', 'enrichment with external data'],
    'result_format': ['JSON', 'CSV', 'HTML', 'XML', 'protobuf', 'plain text'],
    'feature': ['caching', 'logging', 'authentication', 'rate limiting',
                'data validation', 'configuration management', 'error tracking'],
    'problem_goal': ['shortest path', 'maximum flow', 'minimum spanning tree',
                     'closest pair', 'longest palindrome', 'optimal schedule'],
    'constraints': ['O(n log n) time complexity', 'O(1) space complexity',
                    'handling duplicate values', 'supporting concurrent access',
                    'processing streaming data', 'working with limited memory'],
    'target': ['CPU usage', 'memory consumption', 'disk space', 'network traffic',
               'application logs', 'user behavior patterns'],
    'action': ['an alert', 'auto-scaling', 'cleanup routine', 'a backup',
               'a notification', 'a restart'],
    'condition': ['a threshold is exceeded', 'an error occurs', 'a pattern is detected',
                  'a scheduled time is reached', 'a resource is depleted'],
    'pattern': ['Singleton', 'Factory', 'Observer', 'Strategy', 'Command',
                'Decorator', 'Adapter', 'Facade', 'Proxy', 'Chain of Responsibility'],
    'use_case': ['event handling', 'configuration management', 'UI component creation',
                 'data transformation', 'algorithm selection', 'request processing'],
    'topic': ['the CAP theorem', 'functional programming vs OOP', 'microservices architecture',
              'eventual consistency', 'the Raft consensus algorithm', 'map-reduce paradigm',
              'the actor model', 'dependency injection', 'test-driven development',
              'the CQRS pattern', 'event sourcing', 'domain-driven design'],
    'concept_a': ['REST', 'SQL', 'synchronous processing', 'vertical scaling',
                  'monolithic architecture', 'stateful services'],
    'concept_b': ['GraphQL', 'NoSQL', 'asynchronous processing', 'horizontal scaling',
                  'microservices architecture', 'stateless services'],
    'system': ['a distributed database', 'a real-time streaming platform',
               'a content delivery network', 'a container orchestration system',
               'a message queue system', 'a key-value store'],
    'system_type': ['a real-time chat system', 'an e-commerce platform',
                    'a social media feed', 'a file storage service',
                    'a recommendation engine', 'a monitoring dashboard'],
    'technology': ['TCP congestion control', 'HTTP/2 multiplexing', 'TLS handshake',
                   'the Linux scheduler', 'the Go garbage collector',
                   'the V8 JIT compiler', 'the Bitcoin blockchain'],
    'process': ['deploying a web application', 'migrating a database',
                'handling a payment transaction', 'processing a data pipeline',
                'scaling a microservice', 'debugging a production incident'],
    'problem': ['handling millions of concurrent WebSocket connections',
                'processing real-time analytics with sub-second latency',
                'storing and querying petabytes of time-series data',
                'building a globally distributed file system',
                'implementing a fault-tolerant message delivery system'],
}


def generate_code_sample() -> dict:
    """Generate a code-related distillation sample."""
    template = random.choice(CODE_TEMPLATES)
    fill = {}
    for key, values in FILL_VALUES.items():
        if '{' + key + '}' in template:
            fill[key] = random.choice(values)
    fill['lang'] = random.choice(LANGUAGES)
    text = template.format(**fill)
    return {"text": text}


def generate_instruction_sample() -> dict:
    """Generate an instruction-following distillation sample."""
    template = random.choice(INSTRUCTION_TEMPLATES)
    fill = {}
    for key, values in FILL_VALUES.items():
        if '{' + key + '}' in template:
            fill[key] = random.choice(values)
    text = template.format(**fill)
    return {"text": text}


def generate_with_teacher_api(teacher_url: str, api_key: str, prompt: str) -> Optional[str]:
    """Generate text using a teacher model API (OpenAI-compatible)."""
    import requests
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant. Generate high-quality code or explanations."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2048,
        "temperature": 0.7,
    }
    try:
        resp = requests.post(teacher_url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            print(f"  ⚠️ API error: {resp.status_code} {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"  ⚠️ Request failed: {e}")
        return None


def generate_dataset(
    n_samples: int,
    output_path: str,
    teacher_url: Optional[str] = None,
    api_key: Optional[str] = None,
    use_template: bool = True,
) -> None:
    """Generate a dataset of n_samples."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    count = 0
    with open(output_path, 'w') as f:
        while count < n_samples:
            if use_template or not teacher_url:
                # Use template-based generation (fast, no API call)
                if random.random() < 0.6:  # 60% code, 40% instruction
                    sample = generate_code_sample()
                else:
                    sample = generate_instruction_sample()
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                count += 1
                if count % 100 == 0:
                    print(f"  Generated {count}/{n_samples} samples...")
            else:
                # Use teacher model API (slow, high quality)
                prompt = random.choice([
                    "Write a Python function with detailed docstring and type hints.",
                    "Explain a complex computer science concept in simple terms.",
                    "Implement a design pattern with a real-world example.",
                    "Write a code review comment for a pull request.",
                    "Describe how to optimize a slow database query.",
                ])
                text = generate_with_teacher_api(teacher_url, api_key, prompt)
                if text:
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
                    count += 1
                    if count % 10 == 0:
                        print(f"  Generated {count}/{n_samples} samples (via API)...")
                time.sleep(0.5)  # Rate limiting

    print(f"\n✅ Dataset saved to {output_path}")
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   Samples: {n_samples}, Size: {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="TinyLLM Distillation Dataset Generator")
    parser.add_argument('--output', '-o', default='large_dataset.jsonl',
                        help='Output JSONL file path')
    parser.add_argument('--n-samples', '-n', type=int, default=1000,
                        help='Number of samples to generate')
    parser.add_argument('--teacher-url', default=None,
                        help='Teacher API URL (e.g., https://api.deepseek.com/v1/chat/completions)')
    parser.add_argument('--api-key', default=None,
                        help='API key for teacher model (or set DEEPSEEK_API_KEY env var)')
    parser.add_argument('--use-template', action='store_true', default=True,
                        help='Use template-based generation (fast, no API)')
    parser.add_argument('--no-template', dest='use_template', action='store_false',
                        help='Use teacher API for generation (slow, high quality)')

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get('DEEPSEEK_API_KEY')

    print(f"🚀 TinyLLM Distillation Dataset Generator")
    print(f"   Output:   {args.output}")
    print(f"   Samples:  {args.n_samples}")
    print(f"   Method:   {'Template' if args.use_template else 'Teacher API (DeepSeek)'}")
    print()

    generate_dataset(
        n_samples=args.n_samples,
        output_path=args.output,
        teacher_url=args.teacher_url,
        api_key=api_key,
        use_template=args.use_template,
    )


if __name__ == '__main__':
    main()
