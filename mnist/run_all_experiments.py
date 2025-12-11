#!/usr/bin/env python
"""
Master script to run all MNIST experiments and save results in an organized manner.
Results are saved to mnist/results/ with timestamped subdirectories.
"""

import os
import sys
import json
import datetime
import subprocess
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_ROOT = PROJECT_ROOT / "mnist" / "results"

# All experiments to run
EXPERIMENTS = [
    {
        "name": "gradient_rank",
        "module": "mnist.experiments.gradient_rank_experiment",
        "description": "Track gradient rank dynamics with Adam vs Muon"
    },
    {
        "name": "spectral_dynamics", 
        "module": "mnist.experiments.spectral_dynamics_experiment",
        "description": "Full SVD spectrum tracking"
    },
    {
        "name": "ns_transformation",
        "module": "mnist.experiments.ns_transformation_experiment", 
        "description": "Newton-Schulz transformation analysis"
    },
    {
        "name": "component_gradient",
        "module": "mnist.experiments.component_gradient_experiment",
        "description": "Per-component gradient analysis"
    },
    {
        "name": "modular_lr_scaling",
        "module": "mnist.experiments.modular_lr_scaling_experiment",
        "description": "Per-layer learning rate strategies"
    },
    {
        "name": "mnist_ns",
        "module": "mnist.experiments.mnist_ns_experiment",
        "description": "AdamW vs Muon with different NS iterations"
    },
    {
        "name": "optimizer_comparison",
        "module": "mnist.experiments.optimizer_comparison",
        "description": "Systematic optimizer comparison"
    },
    {
        "name": "final_comparison",
        "module": "mnist.experiments.mnist_final_comparison",
        "description": "Final comprehensive comparison"
    },
]


def ensure_results_dir():
    """Create results directory structure."""
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    for exp in EXPERIMENTS:
        (RESULTS_ROOT / exp["name"]).mkdir(exist_ok=True)


def get_experiment_output_dir(exp_name):
    """Get output directory for an experiment with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_ROOT / exp_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_experiment(exp, output_dir):
    """Run a single experiment and capture output."""
    print(f"\n{'='*60}")
    print(f"Running: {exp['name']}")
    print(f"Description: {exp['description']}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*60}\n")
    
    # Change to project root and set environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["EXPERIMENT_OUTPUT_DIR"] = str(output_dir)
    
    # Run the experiment
    cmd = [sys.executable, "-m", exp["module"]]
    
    # Capture output
    log_file = output_dir / "experiment.log"
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout per experiment
        )
        
        # Save output
        with open(log_file, "w") as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"\n{'='*40} STDOUT {'='*40}\n")
            f.write(result.stdout)
            f.write(f"\n{'='*40} STDERR {'='*40}\n")
            f.write(result.stderr)
        
        # Print output
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        success = result.returncode == 0
        if success:
            print(f"✓ {exp['name']} completed successfully")
        else:
            print(f"✗ {exp['name']} failed with code {result.returncode}")
        
        return success, result.stdout
        
    except subprocess.TimeoutExpired:
        print(f"✗ {exp['name']} timed out after 30 minutes")
        return False, "Timeout"
    except Exception as e:
        print(f"✗ {exp['name']} failed with exception: {e}")
        return False, str(e)


def move_generated_files(exp_name, output_dir):
    """Move any generated PNG/JSON files to the output directory."""
    # Check for common output files in the current directory
    for pattern in ["*.png", "*.json", "*.csv"]:
        for f in PROJECT_ROOT.glob(pattern):
            if f.is_file():
                dest = output_dir / f.name
                print(f"Moving {f.name} -> {dest}")
                f.rename(dest)


def create_summary(results):
    """Create a summary JSON of all experiment results."""
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "experiments": results
    }
    summary_file = RESULTS_ROOT / "experiment_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_file}")
    return summary_file


def main():
    """Run all experiments."""
    print("MNIST Experiments Runner")
    print("="*60)
    
    ensure_results_dir()
    
    results = []
    
    for exp in EXPERIMENTS:
        output_dir = get_experiment_output_dir(exp["name"])
        success, output = run_experiment(exp, output_dir)
        
        # Move any generated files
        move_generated_files(exp["name"], output_dir)
        
        results.append({
            "name": exp["name"],
            "success": success,
            "output_dir": str(output_dir),
            "description": exp["description"]
        })
    
    # Create summary
    summary_file = create_summary(results)
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    successful = sum(1 for r in results if r["success"])
    print(f"\nCompleted: {successful}/{len(results)} experiments")
    
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status} {r['name']}: {r['output_dir']}")
    
    print(f"\nResults saved to: {RESULTS_ROOT}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
