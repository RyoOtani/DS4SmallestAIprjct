"""
checkpoint.py — Spot-instance resilient checkpoint manager.

Features:
- Periodic checkpointing with keep-last-N policy
- Preemption signal handling (SIGTERM → save → exit)
- Optional cloud upload (S3/GCS) after each save
- Atomic writes to prevent corruption on interrupt

Usage:
    from training.checkpoint import CheckpointManager
    ckpt = CheckpointManager("./checkpoints", keep_last=3)
    ckpt.save(model, optimizer, step=1000, metrics={"loss": 0.5})
    ckpt.load(model, optimizer)  # returns (step, metrics)
"""
from __future__ import annotations

import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class CheckpointManager:
    """Atomic checkpointing with preemption handling and cloud upload hooks."""

    def __init__(
        self,
        checkpoint_dir: str,
        keep_last: int = 3,
        upload_hook: Optional[callable] = None,
        preemption_signal: str = "SIGTERM",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.upload_hook = upload_hook
        self._saved_steps: list[int] = []
        self._preempted = False

        # Register preemption handler
        if preemption_signal:
            sig = getattr(signal, preemption_signal, None)
            if sig:
                signal.signal(sig, self._preemption_handler)

    def _preemption_handler(self, signum, frame):
        """Called on SIGTERM (spot preemption). Sets flag for graceful exit."""
        print("\n⚠️  Preemption signal received! Will save checkpoint and exit.")
        self._preempted = True

    @property
    def was_preempted(self) -> bool:
        return self._preempted

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save checkpoint atomically (write to temp → rename)."""
        save_dir = self.checkpoint_dir / f"step_{step:07d}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # ── Save model ──
        # Use state_dict for DeepSpeed compatibility
        model_path = save_dir / "model.pt"
        # Atomic write: write to temp file, then rename
        tmp_model = save_dir / ".model.pt.tmp"
        torch.save({"model_state_dict": model.state_dict()}, str(tmp_model))
        tmp_model.rename(model_path)

        # ── Save optimizer ──
        opt_state = {}
        if optimizer is not None:
            opt_state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            opt_state["scheduler_state_dict"] = scheduler.state_dict()
        if opt_state:
            opt_path = save_dir / "optimizer.pt"
            tmp_opt = save_dir / ".optimizer.pt.tmp"
            torch.save(opt_state, str(tmp_opt))
            tmp_opt.rename(opt_path)

        # ── Save metadata ──
        meta = {
            "step": step,
            "timestamp": time.time(),
            "metrics": metrics or {},
        }
        if extra_state:
            meta["extra"] = extra_state
        meta_path = save_dir / "metadata.json"
        tmp_meta = save_dir / ".metadata.json.tmp"
        with open(tmp_meta, "w") as f:
            json.dump(meta, f, indent=2)
        tmp_meta.rename(meta_path)

        # ── Cleanup old checkpoints ──
        self._saved_steps.append(step)
        self._saved_steps.sort()
        while len(self._saved_steps) > self.keep_last:
            old_step = self._saved_steps.pop(0)
            old_dir = self.checkpoint_dir / f"step_{old_step:07d}"
            if old_dir.exists():
                import shutil
                shutil.rmtree(old_dir)

        print(f"💾 Checkpoint saved: step={step}, dir={save_dir}")
        if metrics:
            print(f"   Metrics: {metrics}")

        # ── Upload hook (async-friendly) ──
        if self.upload_hook:
            try:
                self.upload_hook(str(save_dir))
            except Exception as e:
                print(f"⚠️  Upload failed: {e}")

        return save_dir

    def load(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        step: Optional[int] = None,
        map_location: str = "cpu",
    ) -> Tuple[int, Dict[str, float]]:
        """Load latest (or specific) checkpoint."""
        if step is not None:
            load_dir = self.checkpoint_dir / f"step_{step:07d}"
        else:
            # Find latest
            existing = sorted(self.checkpoint_dir.glob("step_*"))
            if not existing:
                raise FileNotFoundError(f"No checkpoints in {self.checkpoint_dir}")
            load_dir = existing[-1]

        model_path = load_dir / "model.pt"
        if model_path.exists():
            ckpt = torch.load(str(model_path), map_location=map_location, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])

        opt_path = load_dir / "optimizer.pt"
        if opt_path.exists() and optimizer is not None:
            opt_state = torch.load(str(opt_path), map_location=map_location, weights_only=False)
            if "optimizer_state_dict" in opt_state:
                optimizer.load_state_dict(opt_state["optimizer_state_dict"])
            if scheduler is not None and "scheduler_state_dict" in opt_state:
                scheduler.load_state_dict(opt_state["scheduler_state_dict"])

        meta_path = load_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {"step": 0, "metrics": {}}

        print(f"📂 Checkpoint loaded: step={meta['step']}, dir={load_dir}")
        return meta["step"], meta["metrics"]

    def list_checkpoints(self) -> list[dict]:
        """List all saved checkpoints with metadata."""
        ckpts = []
        for d in sorted(self.checkpoint_dir.glob("step_*")):
            meta_path = d / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
            else:
                meta = {"step": 0}
            ckpts.append({
                "dir": str(d),
                "step": meta.get("step", 0),
                "timestamp": meta.get("timestamp", 0),
                "metrics": meta.get("metrics", {}),
            })
        return ckpts


# ═══════════════════════════════════════════════════════
# Cloud upload hooks (examples)
# ═══════════════════════════════════════════════════════

def s3_upload_hook(bucket: str, prefix: str = "checkpoints/"):
    """Returns an upload hook that syncs to S3."""
    def _upload(checkpoint_dir: str):
        import subprocess
        subprocess.run(
            ["aws", "s3", "sync", checkpoint_dir,
             f"s3://{bucket}/{prefix}{Path(checkpoint_dir).name}/",
             "--quiet"],
            check=False,
        )
    return _upload


def gcs_upload_hook(bucket: str, prefix: str = "checkpoints/"):
    """Returns an upload hook that syncs to GCS."""
    def _upload(checkpoint_dir: str):
        import subprocess
        subprocess.run(
            ["gsutil", "-m", "rsync", "-r", checkpoint_dir,
             f"gs://{bucket}/{prefix}{Path(checkpoint_dir).name}/"],
            check=False,
        )
    return _upload
