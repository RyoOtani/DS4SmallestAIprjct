"""Training data pipeline for TinyLLM."""

from __future__ import annotations
import torch
from torch.utils.data import Dataset, IterableDataset
import json
import os
from pathlib import Path
from typing import Optional


class TextDataset(Dataset):
    """Simple text dataset from pre-tokenized binary file."""
    
    def __init__(
        self,
        data_path: str,
        seq_len: int,
        vocab_size: int = 65536,
    ):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        if data_path.endswith('.bin'):
            data = torch.from_file(data_path, dtype=torch.int32, byte_order='little')
            self.data = data
        elif data_path.endswith('.jsonl'):
            self.data = self._load_jsonl(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")
        
        self.n_samples = max(0, len(self.data) - seq_len - 1)
    
    def _load_jsonl(self, path: str) -> torch.Tensor:
        tokens = []
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                tokens.extend(item.get('tokens', item.get('token_ids', [])))
        return torch.tensor(tokens, dtype=torch.int32)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> dict:
        x = self.data[idx:idx + self.seq_len].long()
        y = self.data[idx + 1:idx + self.seq_len + 1].long()
        return {"input_ids": x, "labels": y}


class StreamingTextDataset(IterableDataset):
    """Streaming dataset for very large corpora (doesn't load all into memory)."""
    
    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        pattern: str = "*.bin",
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.pattern = pattern
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files = sorted(self.data_dir.glob(self.pattern))
        
        for fpath in files:
            data = torch.from_file(str(fpath), dtype=torch.int32, byte_order='little')
            n_samples = len(data) - self.seq_len - 1
            
            for i in range(0, n_samples, self.seq_len // 2):  # 50% overlap
                x = data[i:i + self.seq_len].long()
                y = data[i + 1:i + self.seq_len + 1].long()
                yield {"input_ids": x, "labels": y}
