"""
Main Experiment Runner

Runs all three Monte Carlo experiments comparing inventory-based and
symmetric market making strategies at different risk aversion levels.

Experiments:
- exp_1: gamma = 0.1 (moderate risk aversion)
- exp_2: gamma = 0.01 (low risk aversion, convergence test)
- exp_3: gamma = 0.5 (high risk aversion, strong variance reduction)
"""

import numpy as np
from pathlib import Path
import pandas as pd

from market_maker_simulation import SimulationParams, MarketMakerSimulator
from statistical_analysis import StrategyComparison
from visualization import (
    plot_profit_histogram,
    plot_inventory_histogram,
    plot_price_path,
    print_summary_table,
    save_results_markdown
)


def run_experiment(
    gamma: float,
    experiment_id: str,
    experiment_title: str,
    results_dir: Path
) -> dict:
    """
    Run a single experiment for a given risk aversion parameter.
    
    Args:
        gamma: Risk aversion parameter
        experiment_id: Unique identifier (e.g., "exp_1")
        experiment_title: Descriptive title
        results_dir: Directory to save results
        
    Returns:
        Dictionary with all results and statistics
    """
    print(f"\n{'#'*80}")
    print(f"# {experiment_title}")
    print(f"# Experiment ID: {experiment_id}")
    print(f"# Risk Aversion γ = {gamma}")
    print(f"{'#'*80}\n")
    
    # Create simulation parameters
    params = SimulationParams(
        s_0=100.0,
        T=1.0,
        sigma=2.0,
        dt=0.005,
        q_0=0,
        X_0=0.0,
        gamma=gamma,
        k=1.5,
        A=140.0,
        n_simulations=1000,
        random_seed=42
    )
    
    # Verify spread calculation against paper targets
    spread = params.spread
    print(f"Computed spread L(γ={gamma}): {spread:.4f}")
    
    # Verify spread matches expected value from paper
    # Note: Paper reports rounded values, actual calculation may differ slightly
    if gamma == 0.1:
        expected_spread = 1.2908  # (2/0.1) * ln(1 + 0.1/1.5)
    elif gamma == 0.01:
        expected_spread = 1.3289  # (2/0.01) * ln(1 + 0.01/1.5)
    elif gamma == 0.5:
        expected_spread = 1.1507  # (2/0.5) * ln(1 + 0.5/1.5)
    else:
        expected_spread = spread
    
    spread_error = abs(spread - expected_spread)
    assert spread_error < 0.002, f"Spread mismatch: computed {spread:.4f}, expected {expected_spread:.4f}"
    print(f"✓ Spread verification passed (error: {spread_error:.6f})\n")
    
    # Create simulator
    simulator = MarketMakerSimulator(params)
    
    # Run Monte Carlo simulation
    results = simulator.run_monte_carlo(store_first_path=True)
    
    # Extract arrays
    profits_inv = results['profits_inv']
    profits_sym = results['profits_sym']
    q_T_inv = results['q_T_inv']
    q_T_sym = results['q_T_sym']
    first_episode = results['first_episode']
    
    # Print summary table
    print_summary_table(results, gamma, spread, title=experiment_title)
    
    # Statistical analysis
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)
    
    # Profit comparison
    profit_comparison = StrategyComparison(
        profits_inv, profits_sym,
        data_name="Terminal Profit",
        n_bootstrap=10000,
        random_seed=42
    )
    profit_comparison.print_summary()
    
    # Inventory comparison
    inventory_comparison = StrategyComparison(
        q_T_inv, q_T_sym,
        data_name="Terminal Inventory",
        n_bootstrap=10000,
        random_seed=42
    )
    inventory_comparison.print_summary()
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    # Profit histogram
    plot_profit_histogram(
        profits_inv, profits_sym, gamma,
        save_path=results_dir / f"histogram_profit_gamma{gamma}.png"
    )
    
    # Inventory histogram
    plot_inventory_histogram(
        q_T_inv, q_T_sym, gamma,
        save_path=results_dir / f"histogram_inventory_gamma{gamma}.png"
    )
    
    # Path plot (only for first experiment or when explicitly requested)
    if first_episode is not None:
        plot_price_path(
            first_episode.S_path,
            first_episode.r_path,
            first_episode.p_ask_path,
            first_episode.p_bid_path,
            first_episode.q_inv_path,
            gamma,
            params.dt,
            save_path=results_dir / f"path_plot_gamma{gamma}.png"
        )
    
    # Save raw data to CSV
    df = pd.DataFrame({
        'profit_inv': profits_inv,
        'profit_sym': profits_sym,
        'q_T_inv': q_T_inv,
        'q_T_sym': q_T_sym
    })
    csv_path = results_dir / f"raw_data_gamma{gamma}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved raw data to {csv_path}")
    
    # Compile results dictionary
    results_dict = {
        'gamma': gamma,
        'spread': spread,
        'params': params,
        'profits_inv': profits_inv,
        'profits_sym': profits_sym,
        'q_T_inv': q_T_inv,
        'q_T_sym': q_T_sym,
        'first_episode': first_episode,
        'profit_comparison': profit_comparison.get_summary_dict(),
        'inventory_comparison': inventory_comparison.get_summary_dict()
    }
    
    # Save results to markdown
    save_results_markdown(
        results_dict,
        gamma,
        spread,
        results_dir / "RESULTS.md",
        experiment_id,
        experiment_title
    )
    
    return results_dict


def main():
    """Run all three experiments."""
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Initialize RESULTS.md
    with open(results_dir / "RESULTS.md", 'w') as f:
        f.write("# Monte Carlo Simulation Results\n\n")
        f.write("## Market Making: Inventory Strategy vs Symmetric Strategy\n\n")
        f.write("This document contains results from three Monte Carlo experiments ")
        f.write("comparing inventory-based and symmetric quoting strategies for market making.\n\n")
        f.write("---\n\n")
    
    # Run experiments
    all_results = {}
    
    # Experiment 1: gamma = 0.1
    all_results['exp_1'] = run_experiment(
        gamma=0.1,
        experiment_id="exp_1",
        experiment_title="Experiment 1: Moderate Risk Aversion (γ = 0.1)",
        results_dir=results_dir
    )
    
    # Experiment 2: gamma = 0.01
    all_results['exp_2'] = run_experiment(
        gamma=0.01,
        experiment_id="exp_2",
        experiment_title="Experiment 2: Low Risk Aversion (γ = 0.01) - Convergence Test",
        results_dir=results_dir
    )
    
    # Experiment 3: gamma = 0.5
    all_results['exp_3'] = run_experiment(
        gamma=0.5,
        experiment_id="exp_3",
        experiment_title="Experiment 3: High Risk Aversion (γ = 0.5) - Strong Variance Reduction",
        results_dir=results_dir
    )
    
    # Print final summary
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nResults saved to: {results_dir.absolute()}")
    print(f"  - RESULTS.md: Comprehensive markdown summary")
    print(f"  - histogram_*.png: Profit and inventory distribution plots")
    print(f"  - path_plot_*.png: Representative price path plots")
    print(f"  - raw_data_*.csv: Raw simulation data")
    print("\n" + "="*80 + "\n")
    
    # Validation summary
    print("VALIDATION AGAINST PAPER TARGETS:")
    print("="*80)
    
    for exp_id, exp_results in all_results.items():
        gamma = exp_results['gamma']
        comp = exp_results['profit_comparison']
        
        print(f"\n{exp_id} (γ = {gamma}):")
        print(f"  Mean(Profit) - Inventory: {comp['mean_inv']:.2f}")
        print(f"  Std(Profit) - Inventory: {comp['std_inv']:.2f}")
        print(f"  Std(Profit) Ratio: {comp['std_inv']/comp['std_sym']:.4f}")
        
        inv_comp = exp_results['inventory_comparison']
        print(f"  Std(q_T) - Inventory: {inv_comp['std_inv']:.2f}")
        print(f"  Std(q_T) Ratio: {inv_comp['std_inv']/inv_comp['std_sym']:.4f}")
    
    print("\n" + "="*80)
    
    return all_results


if __name__ == "__main__":
    results = main()
