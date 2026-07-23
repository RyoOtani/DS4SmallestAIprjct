#!/usr/bin/env python3
"""
TinyLLM Training CLI — launch training with any method on any device.

Usage:
  tinyllm train lora --preset lora-fast --data ./instructions.jsonl
  tinyllm train dpo   --preset dpo-standard --data ./preferences.jsonl
  tinyllm train full  --config xlarge --provider runpod
  tinyllm train methods    # List available training methods
  tinyllm train presets    # List available presets
  tinyllm train recommend  # Recommend method for your task
  tinyllm train mps-info   # Show MPS device info
"""
from __future__ import annotations
import argparse, json, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.training import (
    MPSDetector, MPSTrainer, MPSConfig,
    TrainingMethodRegistry, TRAINING_PRESETS,
    LoRATrainer, InstructionTuner,
)


def cmd_methods(args):
    """List available training methods."""
    methods = TrainingMethodRegistry.list_methods()
    print("\n📚 Available Training Methods:\n")
    print(f"{'Method':12s} {'Memory':12s} {'Quality':20s} Description")
    print("-" * 80)
    for m in methods:
        print(f"{m['id']:12s} {m['memory']:12s} {m['quality']:20s} {m['description']}")


def cmd_presets(args):
    """List available presets."""
    presets = TrainingMethodRegistry.list_presets()
    print(f"\n🎯 Available Presets ({len(presets)}):\n")
    print(f"{'Preset':20s} {'Method':10s} {'LR':>10s} {'Batch':>6s} {'Epochs':>6s} Description")
    print("-" * 90)
    for p in presets:
        print(f"{p['id']:20s} {p['method']:10s} {p['lr']:>10.1e} {p['batch_size']:>6d} {p['epochs']:>6d} {p['description']}")


def cmd_recommend(args):
    """Recommend training method."""
    recs = TrainingMethodRegistry.recommend(args.task, args.hardware)
    print(f"\n💡 Recommendations for '{args.task}' on {args.hardware}:\n")
    for r in recs:
        print(f"  • {r['method']} ({r['preset']}) — {r['reason']}")


def cmd_mps_info(args):
    """Show MPS device information."""
    MPSDetector.print_info()


def cmd_train(args):
    """Launch a training run."""
    preset_id = args.preset
    preset = TrainingMethodRegistry.get_preset(preset_id)
    if not preset:
        print(f"❌ Unknown preset: {preset_id}")
        print(f"   Available: {', '.join(TRAINING_PRESETS)}")
        return

    print(f"\n🚀 Training: {preset.name}")
    print(f"   Method:  {preset.method}")
    print(f"   LR:      {preset.learning_rate}")
    print(f"   Batch:   {preset.batch_size}")
    print(f"   Epochs:  {preset.epochs}")
    print(f"   Desc:    {preset.description}")

    # Determine device
    MPSDetector.print_info()
    use_mps = MPSDetector.is_available() and not args.no_mps

    if use_mps:
        config = MPSConfig(
            use_bfloat16=args.bfloat16,
            use_amp=not args.no_amp,
            gradient_checkpointing=not args.no_grad_ckpt,
            max_steps=args.max_steps,
        )
        trainer = MPSTrainer(config)
        print(f"\n✅ Training on MPS: {MPSDetector.get_device_info()['device_name']}")

        # Load model
        print("   Loading model...")
        from model.config import get_config
        model_cfg = get_config(args.model_scale)
        print(f"   Model: {model_cfg.name}")

        # Apply training method
        if preset.method == "lora":
            print("   Applying LoRA...")
            # Placeholder: model = load_base_model(model_cfg)
            # model, _ = LoRATrainer.apply_lora(model)
            print("   (Model loading requires torch — run in full environment)")
        elif preset.method == "instruct":
            print(f"   Format: {args.instruct_format}")
            if args.data:
                data = InstructionTuner.load_dataset(args.data, format=args.instruct_format)
                print(f"   Loaded {len(data)} instruction samples")

        print(f"\n   Ready to train! (connect data + model to proceed)")

    else:
        # Cloud or CPU training
        if args.provider:
            print(f"\n☁️  Launching on cloud: {args.provider}")
            print(f"   Run: python3 -m agent.cloud.launcher launch")
            print(f"        --provider {args.provider}")
            print(f"        --config {args.model_scale}")
        else:
            print(f"\n💻 Training on CPU (slow — consider --provider for cloud GPU)")

    # Print the equivalent command
    print(f"\n📋 Equivalent CLI command:")
    lr_str = f"{preset.learning_rate:.0e}"
    cmd = f"python3 -m agent.training.cli train --preset {preset_id} --model-scale {args.model_scale}"
    if args.data:
        cmd += f" --data {args.data}"
    if args.provider:
        cmd += f" --provider {args.provider}"
    print(f"   {cmd}")


def main():
    parser = argparse.ArgumentParser(description="TinyLLM Training CLI")
    sub = parser.add_subparsers(dest="command", help="Commands")

    # train
    p = sub.add_parser("train", help="Launch training")
    p.add_argument("--preset","-p",default="lora-fast",help="Training preset")
    p.add_argument("--model-scale","-m",default="small")
    p.add_argument("--data","-d",help="Training data path")
    p.add_argument("--provider",help="Cloud provider (runpod/vast/aws/...)")
    p.add_argument("--instruct-format",default="sharegpt")
    p.add_argument("--max-steps",type=int,default=1000)
    p.add_argument("--bfloat16",action="store_true",default=True)
    p.add_argument("--no-amp",action="store_true")
    p.add_argument("--no-grad-ckpt",action="store_true")
    p.add_argument("--no-mps",action="store_true")
    p.set_defaults(func=cmd_train)

    # methods
    p = sub.add_parser("methods", help="List training methods")
    p.set_defaults(func=cmd_methods)

    # presets
    p = sub.add_parser("presets", help="List training presets")
    p.set_defaults(func=cmd_presets)

    # recommend
    p = sub.add_parser("recommend", help="Recommend training method")
    p.add_argument("--task","-t",default="chat",help="Task type")
    p.add_argument("--hardware",default="auto",help="Hardware (auto/mps/cuda/cpu)")
    p.set_defaults(func=cmd_recommend)

    # mps-info
    p = sub.add_parser("mps-info", help="Show MPS device info")
    p.set_defaults(func=cmd_mps_info)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
