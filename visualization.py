"""
Visualization Module

Provides plotting functions for market making simulation results.

Includes:
- Histogram overlays comparing profit distributions
- Time series plots of prices and inventory
- Summary tables
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def plot_profit_histogram(
    profits_inv: np.ndarray,
    profits_sym: np.ndarray,
    gamma: float,
    save_path: str = None,
    n_bins: int = 40
):
    """
    Plot overlaid histograms of terminal profit distributions.
    
    Args:
        profits_inv: Terminal profits from inventory strategy
        profits_sym: Terminal profits from symmetric strategy
        gamma: Risk aversion parameter (for title)
        save_path: Path to save figure (optional)
        n_bins: Number of histogram bins
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Compute statistics for labels
    mean_inv = np.mean(profits_inv)
    std_inv = np.std(profits_inv, ddof=1)
    mean_sym = np.mean(profits_sym)
    std_sym = np.std(profits_sym, ddof=1)
    
    # Plot histograms with transparency
    ax.hist(
        profits_inv,
        bins=n_bins,
        alpha=0.6,
        color='blue',
        label=f'Inventory Strategy (μ={mean_inv:.2f}, σ={std_inv:.2f})',
        edgecolor='black',
        linewidth=0.5
    )
    
    ax.hist(
        profits_sym,
        bins=n_bins,
        alpha=0.6,
        color='red',
        label=f'Symmetric Strategy (μ={mean_sym:.2f}, σ={std_sym:.2f})',
        edgecolor='black',
        linewidth=0.5
    )
    
    # Add vertical lines for means
    ax.axvline(mean_inv, color='blue', linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(mean_sym, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    # Labels and title
    ax.set_xlabel('Terminal Profit', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax.set_title(
        f'Terminal Profit Distribution Comparison (γ = {gamma})',
        fontsize=14,
        fontweight='bold'
    )
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved histogram to {save_path}")
    
    plt.close()


def plot_inventory_histogram(
    q_T_inv: np.ndarray,
    q_T_sym: np.ndarray,
    gamma: float,
    save_path: str = None,
    n_bins: int = 30
):
    """
    Plot overlaid histograms of terminal inventory distributions.
    
    Args:
        q_T_inv: Terminal inventories from inventory strategy
        q_T_sym: Terminal inventories from symmetric strategy
        gamma: Risk aversion parameter (for title)
        save_path: Path to save figure (optional)
        n_bins: Number of histogram bins
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Compute statistics for labels
    mean_inv = np.mean(q_T_inv)
    std_inv = np.std(q_T_inv, ddof=1)
    mean_sym = np.mean(q_T_sym)
    std_sym = np.std(q_T_sym, ddof=1)
    
    # Plot histograms with transparency
    ax.hist(
        q_T_inv,
        bins=n_bins,
        alpha=0.6,
        color='green',
        label=f'Inventory Strategy (μ={mean_inv:.2f}, σ={std_inv:.2f})',
        edgecolor='black',
        linewidth=0.5
    )
    
    ax.hist(
        q_T_sym,
        bins=n_bins,
        alpha=0.6,
        color='orange',
        label=f'Symmetric Strategy (μ={mean_sym:.2f}, σ={std_sym:.2f})',
        edgecolor='black',
        linewidth=0.5
    )
    
    # Add vertical lines for means
    ax.axvline(mean_inv, color='green', linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(mean_sym, color='orange', linestyle='--', linewidth=2, alpha=0.8)
    
    # Labels and title
    ax.set_xlabel('Terminal Inventory (q_T)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax.set_title(
        f'Terminal Inventory Distribution Comparison (γ = {gamma})',
        fontsize=14,
        fontweight='bold'
    )
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved inventory histogram to {save_path}")
    
    plt.close()


def plot_price_path(
    S_path: np.ndarray,
    r_path: np.ndarray,
    p_ask_path: np.ndarray,
    p_bid_path: np.ndarray,
    q_inv_path: np.ndarray,
    gamma: float,
    dt: float,
    save_path: str = None
):
    """
    Plot time series of mid-price, reservation price, and quotes.
    
    Reproduces Figure 1 style from the paper showing a representative
    single path for the inventory strategy.
    
    Args:
        S_path: Mid-price path
        r_path: Reservation price path
        p_ask_path: Ask price path
        p_bid_path: Bid price path
        q_inv_path: Inventory path
        gamma: Risk aversion parameter (for title)
        dt: Time step
        save_path: Path to save figure (optional)
    """
    time = np.arange(len(S_path)) * dt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Top panel: Prices
    ax1.plot(time, S_path, 'k-', linewidth=2, label='Mid-price $S_t$', alpha=0.7)
    ax1.plot(time, r_path, 'b--', linewidth=2, label='Reservation price $r_t$', alpha=0.8)
    ax1.plot(time, p_ask_path, 'r-', linewidth=1.5, label='Ask price $p^a_t$', alpha=0.6)
    ax1.plot(time, p_bid_path, 'g-', linewidth=1.5, label='Bid price $p^b_t$', alpha=0.6)
    
    ax1.set_ylabel('Price', fontsize=13, fontweight='bold')
    ax1.set_title(
        f'Market Making Inventory Strategy: Price Dynamics (γ = {gamma})',
        fontsize=14,
        fontweight='bold'
    )
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Inventory
    ax2.plot(time, q_inv_path, 'b-', linewidth=2, alpha=0.7)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.fill_between(time, 0, q_inv_path, alpha=0.3, color='blue')
    
    ax2.set_xlabel('Time', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Inventory $q_t$', fontsize=13, fontweight='bold')
    ax2.set_title('Inventory Evolution', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved path plot to {save_path}")
    
    plt.close()


def print_summary_table(
    results: Dict[str, np.ndarray],
    gamma: float,
    spread: float,
    title: str = "Summary Statistics"
):
    """
    Print formatted summary table of results.
    
    Reproduces Table format from the paper.
    
    Args:
        results: Dictionary with profits_inv, profits_sym, q_T_inv, q_T_sym arrays
        gamma: Risk aversion parameter
        spread: Optimal spread L(gamma)
        title: Table title
    """
    profits_inv = results['profits_inv']
    profits_sym = results['profits_sym']
    q_T_inv = results['q_T_inv']
    q_T_sym = results['q_T_sym']
    
    # Compute statistics
    mean_profit_inv = np.mean(profits_inv)
    std_profit_inv = np.std(profits_inv, ddof=1)
    mean_q_T_inv = np.mean(q_T_inv)
    std_q_T_inv = np.std(q_T_inv, ddof=1)
    
    mean_profit_sym = np.mean(profits_sym)
    std_profit_sym = np.std(profits_sym, ddof=1)
    mean_q_T_sym = np.mean(q_T_sym)
    std_q_T_sym = np.std(q_T_sym, ddof=1)
    
    # Print table
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    print(f"Risk Aversion γ = {gamma}")
    print(f"Optimal Spread L(γ) = {spread:.4f}")
    print(f"Number of Episodes = {len(profits_inv)}")
    print(f"{'='*80}")
    
    print(f"\n{'Strategy':<20} {'Mean(Profit)':<15} {'Std(Profit)':<15} {'Mean(q_T)':<15} {'Std(q_T)':<15}")
    print(f"{'-'*80}")
    print(f"{'Inventory':<20} {mean_profit_inv:<15.4f} {std_profit_inv:<15.4f} {mean_q_T_inv:<15.4f} {std_q_T_inv:<15.4f}")
    print(f"{'Symmetric':<20} {mean_profit_sym:<15.4f} {std_profit_sym:<15.4f} {mean_q_T_sym:<15.4f} {std_q_T_sym:<15.4f}")
    print(f"{'-'*80}")
    
    # Compute ratios
    std_profit_ratio = std_profit_inv / std_profit_sym
    std_q_T_ratio = std_q_T_inv / std_q_T_sym
    mean_profit_diff = mean_profit_inv - mean_profit_sym
    
    print(f"\nKey Ratios:")
    print(f"  Std(Profit) ratio (inv/sym): {std_profit_ratio:.4f}")
    print(f"  Std(q_T) ratio (inv/sym): {std_q_T_ratio:.4f}")
    print(f"  Mean(Profit) difference (inv-sym): {mean_profit_diff:.4f}")
    print(f"{'='*80}\n")


def save_results_markdown(
    results_dict: Dict,
    gamma: float,
    spread: float,
    save_path: str,
    experiment_id: str,
    title: str
):
    """
    Save results in markdown format for the RESULTS.md file.
    
    Args:
        results_dict: Dictionary with all results and statistics
        gamma: Risk aversion parameter
        spread: Optimal spread
        save_path: Path to save markdown file
        experiment_id: Experiment identifier (e.g., "exp_1")
        title: Experiment title
    """
    profits_inv = results_dict['profits_inv']
    profits_sym = results_dict['profits_sym']
    q_T_inv = results_dict['q_T_inv']
    q_T_sym = results_dict['q_T_sym']
    
    # Compute statistics
    mean_profit_inv = np.mean(profits_inv)
    std_profit_inv = np.std(profits_inv, ddof=1)
    se_profit_inv = std_profit_inv / np.sqrt(len(profits_inv))
    mean_q_T_inv = np.mean(q_T_inv)
    std_q_T_inv = np.std(q_T_inv, ddof=1)
    se_q_T_inv = std_q_T_inv / np.sqrt(len(q_T_inv))
    
    mean_profit_sym = np.mean(profits_sym)
    std_profit_sym = np.std(profits_sym, ddof=1)
    se_profit_sym = std_profit_sym / np.sqrt(len(profits_sym))
    mean_q_T_sym = np.mean(q_T_sym)
    std_q_T_sym = np.std(q_T_sym, ddof=1)
    se_q_T_sym = std_q_T_sym / np.sqrt(len(q_T_sym))
    
    # Ratios
    std_profit_ratio = std_profit_inv / std_profit_sym
    std_q_T_ratio = std_q_T_inv / std_q_T_sym
    mean_profit_diff = mean_profit_inv - mean_profit_sym
    
    # Create markdown content
    md_content = f"""## {title}

**Experiment ID:** {experiment_id}  
**Risk Aversion (γ):** {gamma}  
**Optimal Spread L(γ):** {spread:.4f}  
**Number of Episodes:** {len(profits_inv)}

### Summary Statistics

| Strategy | Mean(Profit) | Std(Profit) | Mean(q_T) | Std(q_T) |
|----------|--------------|-------------|-----------|----------|
| **Inventory** | {mean_profit_inv:.4f} (SE: {se_profit_inv:.4f}) | {std_profit_inv:.4f} | {mean_q_T_inv:.4f} (SE: {se_q_T_inv:.4f}) | {std_q_T_inv:.4f} |
| **Symmetric** | {mean_profit_sym:.4f} (SE: {se_profit_sym:.4f}) | {std_profit_sym:.4f} | {mean_q_T_sym:.4f} (SE: {se_q_T_sym:.4f}) | {std_q_T_sym:.4f} |

### Key Findings

- **Std(Profit) Reduction:** Inventory strategy has {std_profit_ratio:.4f}x the variance of symmetric strategy
- **Std(q_T) Reduction:** Inventory strategy has {std_q_T_ratio:.4f}x the inventory variance of symmetric strategy
- **Mean Profit Difference:** {mean_profit_diff:.4f} (inventory - symmetric)

### Statistical Tests

"""
    
    # Add statistical test results if available
    if 'profit_comparison' in results_dict:
        comp = results_dict['profit_comparison']
        md_content += f"""
#### Profit Variance Equality Tests

**F-test:**
- F-statistic: {comp['f_test']['F_statistic']:.4f}
- p-value: {comp['f_test']['p_value']:.4e}
- Variance ratio (inv/sym): {comp['f_test']['var_ratio']:.4f}

**Levene Test (Brown-Forsythe):**
- Statistic: {comp['levene_test']['statistic']:.4f}
- p-value: {comp['levene_test']['p_value']:.4e}

**Bootstrap Confidence Intervals (95%, 10,000 resamples):**
- Mean difference (inv - sym): [{comp['mean_diff_ci']['lower_ci']:.4f}, {comp['mean_diff_ci']['upper_ci']:.4f}]
- Std ratio (inv / sym): [{comp['std_ratio_ci']['lower_ci']:.4f}, {comp['std_ratio_ci']['upper_ci']:.4f}]
- Variance ratio (inv / sym): [{comp['var_ratio_ci']['lower_ci']:.4f}, {comp['var_ratio_ci']['upper_ci']:.4f}]
"""
    
    if 'inventory_comparison' in results_dict:
        comp = results_dict['inventory_comparison']
        md_content += f"""
#### Terminal Inventory (q_T) Variance Equality Tests

**F-test:**
- F-statistic: {comp['f_test']['F_statistic']:.4f}
- p-value: {comp['f_test']['p_value']:.4e}
- Variance ratio (inv/sym): {comp['f_test']['var_ratio']:.4f}

**Levene Test (Brown-Forsythe):**
- Statistic: {comp['levene_test']['statistic']:.4f}
- p-value: {comp['levene_test']['p_value']:.4e}

**Bootstrap Confidence Intervals (95%, 10,000 resamples):**
- Mean difference (inv - sym): [{comp['mean_diff_ci']['lower_ci']:.4f}, {comp['mean_diff_ci']['upper_ci']:.4f}]
- Std ratio (inv / sym): [{comp['std_ratio_ci']['lower_ci']:.4f}, {comp['std_ratio_ci']['upper_ci']:.4f}]
- Variance ratio (inv / sym): [{comp['var_ratio_ci']['lower_ci']:.4f}, {comp['var_ratio_ci']['upper_ci']:.4f}]

---

"""
    
    # Write to file (append mode)
    with open(save_path, 'a') as f:
        f.write(md_content)
    
    print(f"Saved results to {save_path}")
