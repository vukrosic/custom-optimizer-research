"""
Analyze token-level routing decisions from a trained dynamic routing model.
Generates visualizations for paper inclusion.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import argparse
import os
import sys
from collections import defaultdict

# Add root to path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from experiments.models import create_model
from experiments.config import get_config

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

def analyze_routing_patterns(text, model, tokenizer, device):
    """Analyze routing patterns for a given text."""
    model.eval()
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Forward pass with routing extraction
    with torch.no_grad():
        outputs = model(input_ids, return_routing=True)
    
    # Extract routing decisions (batch=0)
    # route_layer_X is the hard choice [seq_len, 2] (one-hot)
    r1_hard = outputs['route_layer_1'][0]
    r2_hard = outputs['route_layer_2'][0]
    
    # Extract raw probabilities (batch=0)
    # router_probs is [seq_len, 2, 2] -> dim 2 is layer (0=L1, 1=L2)
    probs = outputs['router_probs'][0] # [seq_len, 2, 2]
    r1_probs_raw = probs[:, 0, :] # [seq_len, 2]
    r2_probs_raw = probs[:, 1, :] # [seq_len, 2]
    
    # Convert to indices (0=GDN, 1=Softmax)
    r1_idx = r1_hard.argmax(dim=-1).cpu().numpy()
    r2_idx = r2_hard.argmax(dim=-1).cpu().numpy()
    
    # Get probabilities as numpy (cast to float32 first for bfloat16 support)
    r1_probs = r1_probs_raw.float().cpu().numpy()
    r2_probs = r2_probs_raw.float().cpu().numpy()
    
    # Decode tokens for cleaner display
    clean_tokens = []
    for i, token in enumerate(tokens):
        clean_token = tokenizer.decode([input_ids[0][i]]).strip()
        if not clean_token:
            clean_token = token
        clean_tokens.append(clean_token)
    
    return {
        'tokens': clean_tokens,
        'layer1_choice': r1_idx,
        'layer2_choice': r2_idx,
        'layer1_probs': r1_probs,
        'layer2_probs': r2_probs,
    }

def visualize_token_routing_heatmap(routing_data, output_path):
    """Create a heatmap showing which layer each token selects."""
    tokens = routing_data['tokens']
    l1_choice = routing_data['layer1_choice']
    l2_choice = routing_data['layer2_choice']
    
    # Create matrix: rows are layers, columns are tokens
    # Value: 0 = GDN (blue), 1 = Softmax (red)
    routing_matrix = np.stack([l1_choice, l2_choice])
    
    fig, ax = plt.subplots(figsize=(max(12, len(tokens) * 0.3), 4))
    
    # Create custom colormap: blue for GDN, red for Softmax
    colors = ['#4da6ff', '#ff4d4d']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    im = ax.imshow(routing_matrix, cmap=cmap, aspect='auto', interpolation='nearest')
    
    # Set ticks
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks([0, 1])
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(['Layer 1', 'Layer 2'], fontsize=11)
    
    # Add grid
    ax.set_xticks(np.arange(len(tokens)) - 0.5, minor=True)
    ax.set_yticks([-.5, 0.5, 1.5], minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(['Linear (GDN)', 'Softmax (Attn)'])
    
    ax.set_title('Token-Level Routing Decisions', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {output_path}")
    plt.close()

def visualize_routing_confidence(routing_data, output_path):
    """Visualize routing confidence (probability) for each token."""
    tokens = routing_data['tokens']
    l1_probs = routing_data['layer1_probs']
    l2_probs = routing_data['layer2_probs']
    
    # Get softmax probabilities
    l1_softmax_prob = l1_probs[:, 1]
    l2_softmax_prob = l2_probs[:, 1]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, len(tokens) * 0.3), 8))
    
    x = np.arange(len(tokens))
    
    # Layer 1
    colors1 = ['#ff4d4d' if p > 0.5 else '#4da6ff' for p in l1_softmax_prob]
    ax1.bar(x, l1_softmax_prob, color=colors1, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Decision Boundary')
    ax1.set_ylabel('Softmax Probability', fontsize=11)
    ax1.set_title('Layer 1: Routing Confidence', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Layer 2
    colors2 = ['#ff4d4d' if p > 0.5 else '#4da6ff' for p in l2_softmax_prob]
    ax2.bar(x, l2_softmax_prob, color=colors2, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Decision Boundary')
    ax2.set_ylabel('Softmax Probability', fontsize=11)
    ax2.set_xlabel('Tokens', fontsize=11)
    ax2.set_title('Layer 2: Routing Confidence', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add legend for colors
    red_patch = mpatches.Patch(color='#ff4d4d', label='Softmax (>0.5)', alpha=0.7)
    blue_patch = mpatches.Patch(color='#4da6ff', label='Linear GDN (<0.5)', alpha=0.7)
    fig.legend(handles=[red_patch, blue_patch], loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved confidence plot to {output_path}")
    plt.close()

def visualize_routing_distribution(routing_data, output_path):
    """Show overall distribution of routing choices."""
    l1_choice = routing_data['layer1_choice']
    l2_choice = routing_data['layer2_choice']
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Layer 1
    l1_counts = np.bincount(l1_choice, minlength=2)
    l1_pct = l1_counts / len(l1_choice) * 100
    
    axes[0].bar(['Linear (GDN)', 'Softmax (Attn)'], l1_pct, 
                color=['#4da6ff', '#ff4d4d'], alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Percentage (%)', fontsize=11)
    axes[0].set_title('Layer 1 Routing Distribution', fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    for i, v in enumerate(l1_pct):
        axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # Layer 2
    l2_counts = np.bincount(l2_choice, minlength=2)
    l2_pct = l2_counts / len(l2_choice) * 100
    
    axes[1].bar(['Linear (GDN)', 'Softmax (Attn)'], l2_pct,
                color=['#4da6ff', '#ff4d4d'], alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Percentage (%)', fontsize=11)
    axes[1].set_title('Layer 2 Routing Distribution', fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    for i, v in enumerate(l2_pct):
        axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved distribution plot to {output_path}")
    plt.close()

def print_routing_table(routing_data):
    """Print a formatted table of routing decisions."""
    tokens = routing_data['tokens']
    l1_choice = routing_data['layer1_choice']
    l2_choice = routing_data['layer2_choice']
    l1_probs = routing_data['layer1_probs']
    l2_probs = routing_data['layer2_probs']
    
    print("\n" + "="*90)
    print(f"{'TOKEN':<20} | {'LAYER 1':<25} | {'LAYER 2':<25}")
    print("="*90)
    
    for i, token in enumerate(tokens):
        l1_name = "SOFTMAX" if l1_choice[i] == 1 else "LINEAR"
        l2_name = "SOFTMAX" if l2_choice[i] == 1 else "LINEAR"
        l1_conf = l1_probs[i, l1_choice[i]]
        l2_conf = l2_probs[i, l2_choice[i]]
        
        print(f"{token:<20} | {l1_name:<10} ({l1_conf:.2f}) | {l2_name:<10} ({l2_conf:.2f})")
    
    print("="*90)
    
    # Print summary statistics
    l1_softmax_pct = (l1_choice == 1).sum() / len(l1_choice) * 100
    l2_softmax_pct = (l2_choice == 1).sum() / len(l2_choice) * 100
    
    print(f"\nSummary:")
    print(f"  Layer 1: {100-l1_softmax_pct:.1f}% Linear (GDN), {l1_softmax_pct:.1f}% Softmax")
    print(f"  Layer 2: {100-l2_softmax_pct:.1f}% Linear (GDN), {l2_softmax_pct:.1f}% Softmax")
    print()

def main():
    parser = argparse.ArgumentParser(description='Analyze token-level routing patterns')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_dynamic/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--text', type=str, 
                        default="The quick brown fox jumps over the lazy dog. However, the complex mathematical intricacies of differential equations and quantum mechanics require deeper analytical understanding.",
                        help='Text to analyze')
    parser.add_argument('--output_dir', type=str, default='routing_analysis',
                        help='Output directory for visualizations')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Please train the model first or specify correct path.")
        return
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    
    # Ensure dynamic routing is enabled
    config.use_dynamic_routing = True
    
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    if device == 'cuda':
        model = model.to(torch.bfloat16)
    print(f"Model loaded on {device}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    
    # Analyze routing
    print(f"\nAnalyzing text: \"{args.text}\"\n")
    routing_data = analyze_routing_patterns(args.text, model, tokenizer, device)
    
    # Print table
    print_routing_table(routing_data)
    
    # Generate visualizations
    print("Generating visualizations...")
    
    visualize_token_routing_heatmap(
        routing_data, 
        os.path.join(args.output_dir, 'token_routing_heatmap.png')
    )
    
    visualize_routing_confidence(
        routing_data,
        os.path.join(args.output_dir, 'routing_confidence.png')
    )
    
    visualize_routing_distribution(
        routing_data,
        os.path.join(args.output_dir, 'routing_distribution.png')
    )
    
    print(f"\n✓ All visualizations saved to {args.output_dir}/")
    print(f"  - token_routing_heatmap.png (shows which layer each token selects)")
    print(f"  - routing_confidence.png (shows routing confidence for each token)")
    print(f"  - routing_distribution.png (shows overall routing statistics)")

if __name__ == "__main__":
    main()
