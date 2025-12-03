"""
Visualize routing patterns for dynamic routing model

This script analyzes which tokens route to GDN vs Softmax
and identifies patterns in the routing decisions.
"""

import torch
import sys
import os
from pathlib import Path

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from experiments.exp11_dynamic_routing.config import get_config
from experiments.exp11_dynamic_routing.models import create_model


@torch.no_grad()
def analyze_routing(checkpoint_path, sample_text="The quick brown fox jumps over the lazy dog"):
    """Analyze routing decisions for sample text"""
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    
    if not config.use_dynamic_routing:
        print("❌ This is not a dynamic routing model!")
        return
    
    # Create model
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("="*70)
    print("Routing Pattern Analysis")
    print("="*70)
    print(f"\nModel: {checkpoint_path}")
    print(f"Sample text: {sample_text}\n")
    
    # TODO: Add tokenization and routing visualization
    # This requires the tokenizer to be available
    print("⚠️  Full implementation requires tokenizer integration")
    print("    For now, check routing statistics in training logs")
    
    # Print routing statistics from model
    if hasattr(model, 'routing_stats'):
        stats = model.routing_stats
        total = stats.get('total_tokens', 1)
        if total > 0:
            print("\nRouting Statistics (from training):")
            print(f"  Layer 1: GDN={100*stats['layer_1_gdn_count']/total:.1f}%, "
                  f"Softmax={100*stats['layer_1_attn_count']/total:.1f}%")
            print(f"  Layer 2: GDN={100*stats['layer_2_gdn_count']/total:.1f}%, "
                  f"Softmax={100*stats['layer_2_attn_count']/total:.1f}%")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to dynamic routing checkpoint')
    parser.add_argument('--text', type=str, default="The quick brown fox",
                        help='Sample text to analyze')
    args = parser.parse_args()
    
    analyze_routing(args.checkpoint, args.text)
