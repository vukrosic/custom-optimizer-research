"""
Training script for Experiment 11: Dynamic Routing vs Static Baseline

Usage:
    # Train baseline (static layers)
    python run_experiment.py --config baseline
    
    # Train dynamic routing
    python run_experiment.py --config dynamic
    
    # Resume from checkpoint
    python run_experiment.py --config dynamic --resume checkpoints_dynamic/best_model.pt
"""

import torch
import torch.nn as nn
import sys
import os
import time
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Fix tokenizer parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from experiments.exp11_dynamic_routing.config import get_config
from experiments.exp11

_dynamic_routing.models import create_model
from data.loader import load_and_cache_data
from utils.helpers import set_seed
from torch.utils.data import DataLoader


class Trainer:
    """Training manager for Dynamic Routing Experiment"""
    
    def __init__(self, model, config, train_loader, val_loader, device, save_dir=None):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else Path("checkpoints")
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # History
        self.train_history = []
        self.val_history = []
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            return max(0.1, (self.config.max_steps - step) / (self.config.max_steps - self.config.warmup_steps))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        
        if isinstance(batch, (list, tuple)):
            input_ids = batch[0].to(self.device)
        else:
            input_ids = batch.to(self.device)
        
        labels = input_ids.clone()
        
        # Forward pass
        outputs = self.model(input_ids=input_ids, labels=labels, step=self.global_step)
        loss = outputs.loss
        
        # Log auxiliary loss for dynamic routing
        aux_loss = None
        if self.config.use_dynamic_routing and outputs.past_key_values:
            aux_loss = outputs.past_key_values[0]
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return loss.item(), aux_loss.item() if aux_loss is not None else 0.0
    
    @torch.no_grad()
    def evaluate(self, max_batches=None):
        """Evaluate on validation set"""
        self.model.eval()
        
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        for i, batch in enumerate(self.val_loader):
            if max_batches and i >= max_batches:
                break
            
            if isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(self.device)
            else:
                input_ids = batch.to(self.device)
            
            labels = input_ids.clone()
            
            outputs = self.model(input_ids=input_ids, labels=labels, step=self.global_step)
            loss = outputs.loss
            logits = outputs.logits
            
            # Calculate accuracy
            predictions = logits.argmax(dim=-1)
            shift_preds = predictions[..., :-1].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            correct = (shift_preds == shift_labels).sum().item()
            
            total_loss += loss.item() * input_ids.numel()
            total_correct += correct
            total_tokens += shift_labels.numel()
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'perplexity': perplexity,
        }
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("Starting Training")
        print("="*70)
        
        start_time = time.time()
        running_loss = 0
        running_aux_loss = 0
        steps_since_log = 0
        
        while self.global_step < self.config.max_steps:
            for batch in self.train_loader:
                if self.global_step >= self.config.max_steps:
                    break
                
                # Training step
                loss, aux_loss = self.train_step(batch)
                running_loss += loss
                running_aux_loss += aux_loss
                steps_since_log += 1
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.log_interval == 0:
                    avg_loss = running_loss / steps_since_log
                    avg_aux = running_aux_loss / steps_since_log
                    lr = self.scheduler.get_last_lr()[0]
                    
                    log_msg = (f"Step {self.global_step}/{self.config.max_steps} | "
                               f"Loss: {avg_loss:.4f} | "
                               f"LR: {lr:.6f}")
                    
                    if self.config.use_dynamic_routing:
                        log_msg += f" | Aux: {avg_aux:.4f}"
                    
                    print(log_msg)
                    
                    self.train_history.append({
                        'step': self.global_step,
                        'loss': avg_loss,
                        'aux_loss': avg_aux if self.config.use_dynamic_routing else 0.0,
                        'lr': lr,
                    })
                    
                    running_loss = 0
                    running_aux_loss = 0
                    steps_since_log = 0
                
                # Evaluation
                if self.global_step % self.config.eval_interval == 0:
                    val_metrics = self.evaluate(max_batches=self.config.eval_batches)
                    
                    print(f"\n{'='*70}")
                    print(f"Evaluation at step {self.global_step}")
                    print(f"{'='*70}")
                    print(f"Val Loss: {val_metrics['loss']:.4f}")
                    print(f"Val Accuracy: {val_metrics['accuracy']:.4f} ({val_metrics['accuracy']*100:.2f}%)")
                    print(f"Val Perplexity: {val_metrics['perplexity']:.2f}")
                    
                    # Log routing stats for dynamic model
                    if self.config.use_dynamic_routing and hasattr(self.model, 'get_routing_stats'):
                        routing_stats = self.model.get_routing_stats(reset=True)
                        print(f"\nRouting Statistics:")
                        print(f"  Layer 1: GDN={routing_stats['layer_1_gdn_pct']:.1f}%, Softmax={routing_stats['layer_1_attn_pct']:.1f}%")
                        print(f"  Layer 2: GDN={routing_stats['layer_2_gdn_pct']:.1f}%, Softmax={routing_stats['layer_2_attn_pct']:.1f}%")
                        val_metrics.update(routing_stats)
                    
                    print(f"{'='*70}\n")
                    
                    self.val_history.append({
                        'step': self.global_step,
                        **val_metrics
                    })
                    
                    # Save best model
                    if val_metrics['loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['loss']
                        self.save_checkpoint('best_model.pt')
                        print(f"✓ New best validation loss: {self.best_val_loss:.4f} (saved)")
        
        total_time = time.time() - start_time
        
        # Save final model
        self.save_checkpoint('final_model.pt')
        
        print(f"\n{'='*70}")
        print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f}m)")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*70}\n")
        
        return {
            'total_time': total_time,
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history,
        }
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint_path = self.save_dir / filename
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'train_history': self.train_history,
            'val_history': self.val_history,
        }
        
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path


def main():
    """Main experiment function"""
    parser = argparse.ArgumentParser(description='Train Dynamic Routing vs Baseline')
    parser.add_argument('--config', type=str, default='baseline',
                        choices=['baseline', 'dynamic', 'dynamic_aggressive'],
                        help='Config to use (baseline or dynamic)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    config = get_config(args.config)
    
    print("="*70)
    print(f"EXPERIMENT 11: Dynamic Routing")
    print(f"Configuration: {args.config}")
    print("="*70)
    
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    
    print(f"\nUsing device: {device}")
    print(f"Architecture: {config.hidden_size}d, {config.num_hidden_layers} layers")
    print(f"Dynamic routing: {config.use_dynamic_routing}")
    if config.use_dynamic_routing:
        print(f"  Routed layers: {config.routed_layers}")
        print(f"  Fixed GDN layers: {config.fixed_gdn_layers}")
        print(f"  Fixed Attn layers: {config.fixed_attn_layers}")
        print(f"  Load balance alpha: {config.load_balance_alpha}")
    
    # Load data
    print("\n" + "="*70)
    print("Loading Data")
    print("="*70)
    
    from dataclasses import dataclass
    @dataclass
    class DataConfig:
        num_documents: int = config.num_documents
        max_tokens: int = config.max_tokens
        vocab_size: int = config.vocab_size
    
    data_config = DataConfig()
    texts, tokenizer, tokens = load_and_cache_data(data_config)
    config.vocab_size = len(tokenizer)
    
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Total tokens: {len(tokens):,}")
    
    # Split tokens
    val_split_ratio = 0.1
    val_token_start = int(len(tokens) * (1 - val_split_ratio))
    
    train_tokens = tokens[:val_token_start]
    val_tokens = tokens[val_token_start:]
    
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")
    
    # Create data loaders
    from data.streaming_dataset import create_progressive_loaders
    
    train_loader, val_loader = create_progressive_loaders(
        train_tokens, val_tokens,
        config.max_seq_len, config.batch_size,
        None, None
    )
    
    print(f"Train batches: {len(train_loader):,}")
    print(f"Val batches: {len(val_loader):,}")
    
    # Create model
    print("\n" + "="*70)
    print("Creating Model")
    print("="*70)
    
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    if dtype == torch.float32:
        print("⚠ Warning: bfloat16 not supported")
    
    model = create_model(config)
    model = model.to(device=device, dtype=dtype)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Using dtype: {dtype}")
    
    # Train
    checkpoints_dir = Path(__file__).parent / config.checkpoint_dir
    results_dir = Path(__file__).parent / f"results_{args.config}"
    
    print(f"\n📁 Output directories:")
    print(f"   Checkpoints: {checkpoints_dir}")
    print(f"   Results: {results_dir}")
    
    trainer = Trainer(model, config, train_loader, val_loader, device, save_dir=checkpoints_dir)
    
    # Load checkpoint if resuming
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.global_step = checkpoint['global_step']
        trainer.best_val_loss = checkpoint['best_val_loss']
        print(f"\n✓ Resumed from checkpoint: {args.resume}")
        print(f"   Step: {trainer.global_step}, Best val loss: {trainer.best_val_loss:.4f}")
    
    results = trainer.train()
    
    # Save results
    results_dir.mkdir(exist_ok=True, parents=True)
    
    results_summary = {
        'experiment': 'exp11_dynamic_routing',
        'config_name': args.config,
        'config': {
            'use_dynamic_routing': config.use_dynamic_routing,
            'routed_layers': config.routed_layers,
            'hidden_size': config.hidden_size,
            'num_layers': config.num_hidden_layers,
            'learning_rate': config.learning_rate,
            'load_balance_alpha': config.load_balance_alpha if config.use_dynamic_routing else 0.0,
        },
        'results': {
            'total_time': results['total_time'],
            'best_val_loss': results['best_val_loss'],
        },
    }
    
    with open(results_dir / 'training_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults saved to: {results_dir / 'training_results.json'}")
    
    print("\n" + "="*70)
    print("✅ EXPERIMENT COMPLETED!")
    print("="*70)
    print(f"Configuration: {args.config}")
    print(f"Best Val Loss: {results['best_val_loss']:.4f}")
    print(f"Training Time: {results['total_time']:.1f}s ({results['total_time']/60:.1f}m)")
    print(f"\n💾 Saved to: {checkpoints_dir}/best_model.pt")
    print("="*70)


if __name__ == "__main__":
    main()
