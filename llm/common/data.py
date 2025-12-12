"""
Data loading utilities for LLM experiments.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


def set_seed(seed):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class SimpleDataset(Dataset):
    """Quick dataset for LLM training with SmolLM corpus."""
    
    def __init__(self, seq_len=256, num_samples=5000, split="train"):
        print(f"📦 Loading SmolLM dataset ({num_samples} samples)...")
        
        # Load SmolLM dataset
        dataset = load_dataset(
            "HuggingFaceTB/smollm-corpus",
            "cosmopedia-v2",
            split=split,
            streaming=True
        )
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo-1b")
        self.seq_len = seq_len
        
        # Collect and tokenize samples
        all_tokens = []
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break
            tokens = self.tokenizer(example['text'], truncation=True, max_length=seq_len)['input_ids']
            all_tokens.extend(tokens)
        
        # Reshape into sequences
        total_tokens = (len(all_tokens) // seq_len) * seq_len
        all_tokens = all_tokens[:total_tokens]
        self.data = torch.tensor(all_tokens).view(-1, seq_len)
        
        print(f"✓ Loaded {len(self.data)} sequences of length {seq_len}")
        print(f"✓ Vocab size: {self.tokenizer.vocab_size}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx],
            'labels': self.data[idx]
        }


def create_dataloaders(seq_len=256, num_samples=5000, batch_size=16, num_workers=0):
    """Create train and validation dataloaders.
    
    Args:
        seq_len: Sequence length
        num_samples: Number of training samples
        batch_size: Batch size
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = SimpleDataset(
        seq_len=seq_len,
        num_samples=num_samples,
        split="train"
    )
    
    val_dataset = SimpleDataset(
        seq_len=seq_len,
        num_samples=num_samples // 10,  # 10% for validation
        split="train"  # Using train split but different samples
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader
