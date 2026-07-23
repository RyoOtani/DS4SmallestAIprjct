#!/usr/bin/env python3
"""
Hugging Face Hub Push Script for TinyLLM.

Usage:
  hf auth login                              # First: authenticate
  python3 push_to_hub.py                     # Push all 9 models
  python3 push_to_hub.py --scale nano        # Push single model
  python3 push_to_hub.py --dry-run           # Preview without pushing
"""

import json
import os
import sys
import argparse
from pathlib import Path

# Check if huggingface_hub is available
try:
    from huggingface_hub import HfApi, create_repo, upload_file, login
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("❌ huggingface-hub not installed. Run: pip install huggingface-hub")
    sys.exit(1)

HF_NAMESPACE = "RyoOtani"
BASE_DIR = Path(__file__).parent / "hf_models"

SCALES = ["nano", "small", "medium", "large", "dense-7b", "xlarge", "xxlarge", "mega", "giga"]

SCALE_INFO = {
    "nano":      {"total": "1.5B", "active": "0.5B",  "gguf_q4": "~900 MB"},
    "small":     {"total": "3.0B", "active": "1.0B",  "gguf_q4": "~1.8 GB"},
    "medium":    {"total": "14.5B","active": "5.5B",  "gguf_q4": "~8.5 GB"},
    "large":     {"total": "28.1B","active": "10.8B", "gguf_q4": "~16 GB"},
    "dense-7b":  {"total": "7.2B", "active": "7.2B",  "gguf_q4": "~4.2 GB"},
    "xlarge":    {"total": "43.9B","active": "16.5B", "gguf_q4": "~25 GB"},
    "xxlarge":   {"total": "72.0B","active": "27.0B", "gguf_q4": "~41 GB"},
    "mega":      {"total": "178.6B","active":"67.8B", "gguf_q4": "~102 GB"},
    "giga":      {"total": "6.7T", "active":"314.6B","gguf_q4": "~3.8 TB"},
}


def push_model(api: HfApi, scale: str, namespace: str, dry_run: bool = False):
    """Push a single model's config and model card to Hugging Face Hub."""
    repo_id = f"{namespace}/tinyllm-{scale}"
    model_dir = BASE_DIR / f"tinyllm-{scale}"

    if not model_dir.exists():
        print(f"  ⚠️  {scale}: directory not found, skipping")
        return False

    print(f"\n{'='*60}")
    print(f"📦 {scale:12s} → {repo_id}")
    print(f"   Total: {SCALE_INFO[scale]['total']:>6s} | Active: {SCALE_INFO[scale]['active']:>6s} | GGUF Q4: {SCALE_INFO[scale]['gguf_q4']}")
    print(f"{'='*60}")

    if dry_run:
        print("  [DRY RUN] Would create repo and upload:")
        for f in sorted(model_dir.iterdir()):
            print(f"    - {f.name} ({f.stat().st_size:,} bytes)")
        return True

    try:
        # Create or get repo
        try:
            repo_url = create_repo(
                repo_id=repo_id,
                repo_type="model",
                exist_ok=True,
                private=False,
            )
            print(f"  ✅ Repo ready: {repo_url}")
        except HfHubHTTPError as e:
            if "already exists" in str(e) or "409" in str(e):
                print(f"  ✅ Repo already exists")
            else:
                raise

        # Upload files
        for f in sorted(model_dir.iterdir()):
            if f.is_file():
                print(f"  ⬆️  Uploading: {f.name} ({f.stat().st_size:,} bytes)")
                upload_file(
                    path_or_fileobj=str(f),
                    path_in_repo=f.name,
                    repo_id=repo_id,
                    repo_type="model",
                )

        print(f"  🎉 Successfully pushed to https://huggingface.co/{repo_id}")
        return True

    except HfHubHTTPError as e:
        print(f"  ❌ HF Hub error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Push TinyLLM models to Hugging Face Hub")
    parser.add_argument("--scale", "-s", type=str, help="Push a single scale (e.g., nano)")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Preview without pushing")
    parser.add_argument("--namespace", type=str, default=HF_NAMESPACE, help="HF namespace")
    args = parser.parse_args()

    namespace = args.namespace

    # Check auth
    api = HfApi()
    try:
        whoami = api.whoami()
        print(f"✅ Logged in as: {whoami.get('name', 'unknown')}")
    except Exception:
        print("❌ Not logged in to Hugging Face!")
        print("   Run: hf auth login")
        print("   Or set HF_TOKEN environment variable.")
        sys.exit(1)

    # Push models
    scales_to_push = [args.scale] if args.scale else SCALES

    results = {}
    for scale in scales_to_push:
        results[scale] = push_model(api, scale, namespace, dry_run=args.dry_run)

    # Summary
    print(f"\n{'='*60}")
    print("📊 Summary:")
    for scale, ok in results.items():
        status = "✅" if ok else "❌"
        url = f"https://huggingface.co/{namespace}/tinyllm-{scale}"
        print(f"  {status} {scale:12s} → {url}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
