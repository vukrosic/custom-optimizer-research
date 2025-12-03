"""
Compare baseline vs dynamic routing results
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_results(config_name):
    """Load results for a configuration"""
    results_dir = Path(__file__).parent / f"results_{config_name}"
    results_file = results_dir / "training_results.json"
    
    if not results_file.exists():
        print(f"⚠️  Results not found for {config_name}: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)


def main():
    print("="*70)
    print("Comparing Baseline vs Dynamic Routing")
    print("="*70)
    
    # Load results
    baseline_results = load_results('baseline')
    dynamic_results = load_results('dynamic')
    
    if baseline_results is None or dynamic_results is None:
        print("\n❌ Cannot compare - missing results. Train both models first:")
        if baseline_results is None:
            print("   python run_experiment.py --config baseline")
        if dynamic_results is None:
            print("   python run_experiment.py --config dynamic")
        return
    
    # Compare
    print("\n📊 Results Comparison:\n")
    
    baseline_loss = baseline_results['results']['best_val_loss']
    dynamic_loss = dynamic_results['results']['best_val_loss']
    
    print(f"{'Configuration':<20} {'Best Val Loss':<15} {'Training Time':<15}")
    print("-" * 50)
    print(f"{'Baseline (Static)':<20} {baseline_loss:<15.4f} {baseline_results['results']['total_time']:<15.1f}")
    print(f"{'Dynamic Routing':<20} {dynamic_loss:<15.4f} {dynamic_results['results']['total_time']:<15.1f}")
    
    # Winner
    print("\n" + "="*70)
    if dynamic_loss < baseline_loss:
        improvement = ((baseline_loss - dynamic_loss) / baseline_loss) * 100
        print(f"🏆 WINNER: Dynamic Routing")
        print(f"   Improvement: {improvement:.2f}% better than baseline")
        print(f"   Dynamic routing successfully learned better token-specific choices!")
    elif baseline_loss < dynamic_loss:
        regression = ((dynamic_loss - baseline_loss) / baseline_loss) * 100
        print(f"🏆 WINNER: Baseline (Static)")
        print(f"   Baseline is {regression:.2f}% better")
        print(f"   Routing overhead may not be worth the complexity")
        print(f"   Try increasing load_balance_alpha to prevent collapse")
    else:
        print(f"🤝 TIE: Both configurations perform equally")
    
    print("="*70)
    
    # Routing statistics (if available for dynamic)
    if 'load_balance_alpha' in dynamic_results['config']:
        print(f"\nDynamic Routing Config:")
        print(f"   Load balance alpha: {dynamic_results['config']['load_balance_alpha']}")
        print(f"   Routed layers: {dynamic_results['config']['routed_layers']}")


if __name__ == "__main__":
    main()
