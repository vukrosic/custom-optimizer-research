"""
Run experiments for modular optimizer research.

Usage:
    # Run single experiment
    python experiments/run_experiments.py --exp baseline
    
    # Run all experiments
    python experiments/run_experiments.py --all
    
    # Quick test run
    python experiments/run_experiments.py --exp baseline --max_steps 50
"""

import argparse
import json
import math
import time
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ModelConfig, TrainingConfig
from model import create_model
from train import SmolLMDataset, set_seed
from experiments.experiment_config import get_experiment_configs, get_experiment, ExperimentConfig
from experiments.modular_optimizer import ModularOptimizer, ModularScheduler


class ExperimentLogger:
    """Logs training metrics to JSON for later analysis."""
    
    def __init__(self, experiment_name: str, output_dir: Path):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'steps': [],
            'config': None,
        }
        
    def log_config(self, config: ExperimentConfig):
        """Log experiment configuration."""
        self.metrics['config'] = {
            'name': config.name,
            'description': config.description,
            'max_steps': config.max_steps,
            'batch_size': config.batch_size,
            'hidden_size': config.hidden_size,
            'num_layers': config.num_layers,
            'embedding_optimizer': config.embedding_optimizer.name,
            'embedding_lr': config.embedding_optimizer.lr,
            'attention_optimizer': config.attention_optimizer.name,
            'attention_lr': config.attention_optimizer.lr,
            'ffn_optimizer': config.ffn_optimizer.name,
            'ffn_lr': config.ffn_optimizer.lr,
        }
    
    def log_step(self, step: int, metrics: Dict[str, float]):
        """Log metrics for a training step."""
        entry = {'step': step, **metrics}
        self.metrics['steps'].append(entry)
    
    def save(self):
        """Save metrics to JSON file."""
        self.metrics['end_time'] = datetime.now().isoformat()
        
        output_file = self.output_dir / f"{self.experiment_name}_metrics.json"
        with open(output_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"📊 Saved metrics to {output_file}")
        return output_file


def compute_weight_stats(model: torch.nn.Module) -> Dict[str, float]:
    """Compute weight statistics for different parameter groups."""
    stats = {}
    
    group_norms = {
        'embedding': [],
        'attention': [],
        'ffn': [],
        'norm': [],
    }
    
    for name, param in model.named_parameters():
        if 'embed' in name.lower() or 'lm_head' in name.lower():
            group = 'embedding'
        elif 'attention' in name.lower() or 'qkv' in name.lower():
            group = 'attention'
        elif 'feed_forward' in name.lower() or 'w1' in name or 'w2' in name or 'w3' in name:
            group = 'ffn'
        elif 'norm' in name.lower():
            group = 'norm'
        else:
            continue
        
        # Frobenius norm
        norm = param.data.norm().item()
        group_norms[group].append(norm)
        
        # Spectral norm for 2D matrices
        if param.ndim == 2:
            try:
                # Approximate spectral norm (largest singular value)
                # Use power iteration for efficiency
                u = torch.randn(param.size(0), device=param.device)
                for _ in range(3):
                    v = param.t() @ u
                    v = v / v.norm()
                    u = param @ v
                    u = u / u.norm()
                spectral = (u @ param @ v).item()
                stats[f'{name}_spectral'] = abs(spectral)
            except:
                pass
    
    # Aggregate stats per group
    for group, norms in group_norms.items():
        if norms:
            stats[f'{group}_frob_mean'] = sum(norms) / len(norms)
            stats[f'{group}_frob_max'] = max(norms)
    
    return stats


def run_experiment(config: ExperimentConfig, output_dir: Path) -> Dict[str, Any]:
    """Run a single experiment and return results."""
    
    print(f"\n{'='*60}")
    print(f"🧪 Running Experiment: {config.name}")
    print(f"   {config.description}")
    print(f"{'='*60}\n")
    
    # Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on {device}")
    
    # Logger
    logger = ExperimentLogger(config.name, output_dir)
    logger.log_config(config)
    
    # Data
    print(f"\n📦 Loading dataset...")
    dataset = SmolLMDataset(
        seq_len=config.max_seq_len,
        num_samples=config.num_samples,
    )
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # Model
    model_config = ModelConfig(
        vocab_size=dataset.tokenizer.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        max_seq_len=config.max_seq_len,
    )
    
    model = create_model(model_config).to(device)
    
    # Modular Optimizer
    print("\n🔧 Setting up modular optimizer...")
    optimizer = ModularOptimizer(model, config)
    scheduler = ModularScheduler(optimizer, config.warmup_steps, config.max_steps)
    
    # Training loop
    print(f"\n🏋️ Starting training for {config.max_steps} steps...\n")
    
    model.train()
    train_iter = iter(train_loader)
    running_loss = 0.0
    start_time = time.time()
    
    for step in range(1, config.max_steps + 1):
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        
        # Logging
        if step % config.log_interval == 0:
            avg_loss = running_loss / config.log_interval
            elapsed = time.time() - start_time
            tokens_per_sec = (config.log_interval * config.batch_size * config.max_seq_len) / elapsed
            lrs = optimizer.get_lr()
            
            # Compute weight stats occasionally
            if step % (config.log_interval * 5) == 0:
                weight_stats = compute_weight_stats(model)
            else:
                weight_stats = {}
            
            # Log metrics
            metrics = {
                'train_loss': avg_loss,
                'tokens_per_sec': tokens_per_sec,
                **{f'lr_{k}': v for k, v in lrs.items()},
                **weight_stats,
            }
            logger.log_step(step, metrics)
            
            lr_str = ", ".join([f"{k}:{v:.2e}" for k, v in list(lrs.items())[:2]])
            print(f"Step {step:5d} | Loss: {avg_loss:.4f} | {tokens_per_sec:.0f} tok/s | LR: {lr_str}")
            
            running_loss = 0.0
            start_time = time.time()
        
        # Evaluation
        if step % config.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device, max_batches=10)
            perplexity = math.exp(val_loss)
            print(f"         | Val Loss: {val_loss:.4f} | Perplexity: {perplexity:.2f}")
            
            # Log validation metrics
            logger.log_step(step, {
                'val_loss': val_loss,
                'perplexity': perplexity,
            })
            model.train()
    
    # Final evaluation
    final_val_loss = evaluate(model, val_loader, device)
    final_perplexity = math.exp(final_val_loss)
    
    # Save results
    logger.metrics['final_val_loss'] = final_val_loss
    logger.metrics['final_perplexity'] = final_perplexity
    logger.save()
    
    # Save model checkpoint
    ckpt_path = output_dir / f"{config.name}_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model_config,
        'experiment_config': config.name,
    }, ckpt_path)
    print(f"💾 Saved model to {ckpt_path}")
    
    print(f"\n✅ Experiment {config.name} complete!")
    print(f"   Final Val Loss: {final_val_loss:.4f}")
    print(f"   Final Perplexity: {final_perplexity:.2f}")
    
    return {
        'name': config.name,
        'final_val_loss': final_val_loss,
        'final_perplexity': final_perplexity,
    }


def evaluate(model, val_loader, device, max_batches=None):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, labels=labels)
            total_loss += outputs["loss"].item()
            num_batches += 1
            
            if max_batches and num_batches >= max_batches:
                break
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Run modular optimizer experiments")
    parser.add_argument("--exp", type=str, default=None, 
                       help="Single experiment to run (baseline, full_muon, attention_muon, ffn_muon, qkv_muon, spherical_embed)")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max steps")
    parser.add_argument("--output_dir", type=str, default="experiment_results", help="Output directory")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.all:
        experiments = get_experiment_configs()
        results = []
        
        for name, config in experiments.items():
            if args.max_steps:
                config.max_steps = args.max_steps
            result = run_experiment(config, output_dir)
            results.append(result)
        
        # Summary
        print("\n" + "="*60)
        print("📊 EXPERIMENT SUMMARY")
        print("="*60)
        for r in sorted(results, key=lambda x: x['final_val_loss']):
            print(f"  {r['name']:20s} | Loss: {r['final_val_loss']:.4f} | PPL: {r['final_perplexity']:.2f}")
        
        # Save summary
        summary_path = output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📊 Saved summary to {summary_path}")
        
    elif args.exp:
        config = get_experiment(args.exp)
        if args.max_steps:
            config.max_steps = args.max_steps
        run_experiment(config, output_dir)
    else:
        print("Please specify --exp <name> or --all")
        print("Available experiments:", list(get_experiment_configs().keys()))


if __name__ == "__main__":
    main()
