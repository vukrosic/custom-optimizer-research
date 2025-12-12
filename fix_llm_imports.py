#!/usr/bin/env python3
"""
Script to fix imports in all copied LLM experiments.
Replaces MNIST-specific imports with LLM equivalents.
"""

import os
import re

# Directory containing experiments
EXPERIMENTS_DIR = "llm/experiments"

# Files to process (exclude already created ones)
FILES_TO_FIX = [
    "llm_ns_transformation_experiment.py",
    "llm_component_gradient_experiment.py",
    "llm_modular_lr_scaling_experiment.py",
    "llm_ns_experiment.py",
    "llm_modular_optimizer_experiment.py",
    "llm_optimizer_comparison.py",
]

def fix_imports(filepath):
    """Fix imports in a Python file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace MNIST imports with LLM imports
    replacements = [
        # Remove MNIST dataset imports
        (r'from torchvision import datasets, transforms\n', ''),
        # Replace mnist.common imports
        (r'from mnist\.common import (.+)', r'from llm.common import \1'),
        # Replace model creation
        (r'class MNISTNet\(nn\.Module\):.+?(?=\n\nclass |\n\ndef |$)', 
         '# Using GPTModel from llm.models.model\n', re.DOTALL),
        # Replace MNIST data loading
        (r'def get_mnist_loaders\(.+?\n    return train_loader, test_loader\n', 
         '# Using create_dataloaders from llm.common\n', re.DOTALL),
        # Update docstrings
        (r'MNIST', 'LLM'),
        (r'mnist', 'llm'),
        # Update file references
        (r'mnist/experiments', 'llm/experiments'),
        (r'mnist/figures', 'llm/figures'),
    ]
    
    for pattern, replacement, *flags in replacements:
        if flags:
            content = re.sub(pattern, replacement, content, flags=flags[0])
        else:
            content = re.sub(pattern, replacement, content)
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"✓ Fixed {filepath}")


def main():
    for filename in FILES_TO_FIX:
        filepath = os.path.join(EXPERIMENTS_DIR, filename)
        if os.path.exists(filepath):
            try:
                fix_imports(filepath)
            except Exception as e:
                print(f"✗ Error fixing {filename}: {e}")
        else:
            print(f"✗ File not found: {filepath}")


if __name__ == '__main__':
    main()
