"""
Cloud GPU Launcher — Provider-agnostic training launch.

Usage:
  tinyllm cloud launch --provider runpod --config xlarge
  tinyllm cloud launch --provider vast    --config medium --spot
  tinyllm cloud compare --config medium   # Compare prices across providers
  tinyllm cloud status  --provider aws
  tinyllm cloud destroy --provider runpod --instance xxx
"""
from __future__ import annotations
import argparse, json, sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.cloud.provider import (
    CloudProvider, LaunchConfig, GPURequirement, GPUType, CostEstimate,
)
from agent.cloud.providers import (
    RunPodProvider, VastAIProvider, GPUSorobanProvider,
    LambdaLabsProvider, AWSProvider, AzureProvider, GCPProvider,
)

PROVIDER_MAP = {
    "runpod": RunPodProvider,
    "vast": VastAIProvider,
    "vastai": VastAIProvider,
    "soroban": GPUSorobanProvider,
    "gpumoroban": GPUSorobanProvider,
    "lambda": LambdaLabsProvider,
    "lambdalabs": LambdaLabsProvider,
    "aws": AWSProvider,
    "azure": AzureProvider,
    "gcp": GCPProvider,
}

def load_config() -> dict:
    """Load cloud config from ~/.tinyllm/cloud.yaml or env."""
    config = {}
    config_path = Path.home() / ".tinyllm" / "cloud.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
    # Override from env
    for key in ["runpod_api_key","vastai_api_key","soroban_api_key","lambda_api_key","aws_ami","azure_rg","gcp_zone"]:
        env_val = os.environ.get(key.upper(), os.environ.get(f"TINYLLM_{key.upper()}",""))
        if env_val:
            config[key] = env_val
    return config

def get_provider(name: str) -> CloudProvider:
    config = load_config()
    provider_cls = PROVIDER_MAP.get(name.lower())
    if not provider_cls:
        print(f"Unknown provider: {name}. Available: {', '.join(PROVIDER_MAP)}")
        sys.exit(1)
    return provider_cls(config=config)

def cmd_launch(args):
    """Launch a training job on a cloud GPU provider."""
    provider = get_provider(args.provider)
    gpu = GPURequirement(
        gpu_type=GPUType(args.gpu_type),
        count=args.gpus,
        min_vram_per_gpu_gb=args.min_vram,
    )
    config = LaunchConfig(
        model_scale=args.config,
        gpu_requirement=gpu,
        use_spot=not args.no_spot,
        max_price_per_hour=args.max_price,
        auto_shutdown_hours=args.timeout,
        keep_alive=args.keep_alive,
        docker_image=args.image,
    )

    # Show cost estimate
    estimate = provider.estimate_cost(config)
    print(f"\n{estimate.summary()}")
    if not args.yes:
        confirm = input("\nLaunch? [y/N] ")
        if confirm.lower() not in ("y","yes"):
            print("Cancelled.")
            return

    # Launch
    print(f"\n🚀 Launching on {provider.NAME}...")
    instance = provider.launch(config)
    if instance.status.value == "error":
        print(f"❌ Failed: {instance.metadata.get('error','Unknown error')}")
        return

    print(f"✅ Instance created: {instance.instance_id}")
    print(f"   Provider: {instance.provider}")
    print(f"   GPUs: {instance.gpu_count}x {instance.gpu_type}")
    print(f"   ~${instance.price_per_hour:.2f}/hr")
    if instance.ssh_host:
        print(f"   SSH: ssh {instance.ssh_host} -p {instance.ssh_port}")

    # Wait until ready
    if not args.no_wait:
        print("\n⏳ Waiting for instance to be ready...")
        ready = provider.wait_until_ready(instance.instance_id, on_status=lambda s: print(f"   Status: {s.value}"))
        if ready:
            print("✅ Instance is RUNNING!")
        else:
            print("⚠️  Timed out waiting for instance.")

def cmd_status(args):
    """Show status of cloud instances."""
    provider = get_provider(args.provider)
    instances = provider.list_instances()
    print(f"\n📊 {provider.NAME} Instances ({len(instances)}):")
    print("-" * 80)
    for inst in instances:
        status_icon = {"running":"🟢","provisioning":"🟡","stopped":"🔴","terminated":"⚫","error":"💥"}.get(inst.status.value,"❓")
        print(f"{status_icon} {inst.instance_id[:20]:20s} | {inst.instance_name[:30]:30s} | {inst.gpu_count}x {inst.gpu_type:15s} | ${inst.price_per_hour:.2f}/hr | {inst.status.value}")

def cmd_destroy(args):
    """Destroy a cloud instance."""
    provider = get_provider(args.provider)
    print(f"\n🗑️  Destroying {args.instance} on {provider.NAME}...")
    if provider.destroy(args.instance):
        print("✅ Destroyed.")
    else:
        print("❌ Failed to destroy.")

def cmd_compare(args):
    """Compare prices across all providers for a given config."""
    gpu = GPURequirement(gpu_type=GPUType(args.gpu_type), count=args.gpus)
    config = LaunchConfig(model_scale=args.config, gpu_requirement=gpu, use_spot=not args.no_spot, auto_shutdown_hours=args.timeout)

    print(f"\n💰 Price Comparison: {args.gpus}x {args.gpu_type} ({args.config})")
    print("-" * 80)
    print(f"{'Provider':15s} {'On-Demand/hr':>14s} {'Spot/hr':>14s} {'Est. Total':>14s} {'Est. Spot':>14s}")
    print("-" * 80)

    estimates = []
    for name in ["runpod","vastai","soroban","lambda","aws","azure","gcp"]:
        try:
            provider = get_provider(name)
            est = provider.estimate_cost(config)
            estimates.append(est)
            print(f"{est.provider:15s} ${est.price_per_hour_total:>13.2f} ${est.price_per_hour_total*est.spot_discount_pct/100:>13.2f} ${est.estimated_total_cost:>13.0f} ${est.spot_price_total:>13.0f}")
        except Exception:
            pass

    if estimates:
        best = min(estimates, key=lambda e: e.spot_price_total)
        print("-" * 80)
        print(f"🏆 Best deal: {best.provider} (${best.spot_price_total:.0f} spot)")

def cmd_prices(args):
    """List available GPU types and prices."""
    provider = get_provider(args.provider)
    gpus = provider.get_available_gpus()
    print(f"\n🖥️  {provider.NAME} Available GPUs:")
    print("-" * 50)
    for gpu in gpus:
        spot = gpu.get("spot_price", 0)
        print(f"  {gpu['type']:20s} ${gpu['price_per_hour']:.2f}/hr" + (f" (spot: ${spot:.2f}/hr)" if spot else ""))

def main():
    parser = argparse.ArgumentParser(description="TinyLLM Cloud GPU Launcher")
    sub = parser.add_subparsers(dest="command", help="Commands")

    # launch
    p = sub.add_parser("launch", help="Launch training on cloud GPU")
    p.add_argument("--provider","-p",required=True,help="Cloud provider")
    p.add_argument("--config","-c",default="medium",help="Model scale")
    p.add_argument("--gpus","-g",type=int,default=8,help="Number of GPUs")
    p.add_argument("--gpu-type","-t",default="a100-80gb",help="GPU type")
    p.add_argument("--min-vram",type=int,default=40,help="Min VRAM per GPU (GB)")
    p.add_argument("--max-price",type=float,default=50.0,help="Max price/hr")
    p.add_argument("--timeout",type=float,default=24.0,help="Auto-shutdown hours")
    p.add_argument("--image",default="pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel")
    p.add_argument("--no-spot",action="store_true")
    p.add_argument("--keep-alive",action="store_true")
    p.add_argument("--no-wait",action="store_true")
    p.add_argument("--yes","-y",action="store_true",help="Skip confirmation")
    p.set_defaults(func=cmd_launch)

    # status
    p = sub.add_parser("status", help="Show cloud instance status")
    p.add_argument("--provider","-p",required=True)
    p.set_defaults(func=cmd_status)

    # destroy
    p = sub.add_parser("destroy", help="Destroy a cloud instance")
    p.add_argument("--provider","-p",required=True)
    p.add_argument("--instance","-i",required=True)
    p.set_defaults(func=cmd_destroy)

    # compare
    p = sub.add_parser("compare", help="Compare prices across providers")
    p.add_argument("--config","-c",default="medium")
    p.add_argument("--gpus","-g",type=int,default=8)
    p.add_argument("--gpu-type","-t",default="a100-80gb")
    p.add_argument("--timeout",type=float,default=24.0)
    p.add_argument("--no-spot",action="store_true")
    p.set_defaults(func=cmd_compare)

    # prices
    p = sub.add_parser("prices", help="List GPU prices for a provider")
    p.add_argument("--provider","-p",required=True)
    p.set_defaults(func=cmd_prices)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    args.func(args)

if __name__ == "__main__":
    main()
