"""
Statistical Analysis Module

Provides statistical tests and bootstrap methods for comparing
inventory-based and symmetric market making strategies.

Includes:
- F-test for variance equality
- Levene and Brown-Forsythe tests for variance equality
- Bootstrap confidence intervals for mean differences and variance ratios
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple


def f_test_variance_equality(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Perform F-test for equality of variances.
    
    H0: Var(x) = Var(y)
    H1: Var(x) != Var(y)
    
    Args:
        x: First sample array
        y: Second sample array
        
    Returns:
        Dictionary with F-statistic, p-value, and variance ratio
    """
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    
    # F-statistic: ratio of variances (larger / smaller for two-tailed test)
    if var_x >= var_y:
        F = var_x / var_y
        df1 = len(x) - 1
        df2 = len(y) - 1
    else:
        F = var_y / var_x
        df1 = len(y) - 1
        df2 = len(x) - 1
    
    # Two-tailed p-value
    p_value = 2 * min(stats.f.cdf(F, df1, df2), 1 - stats.f.cdf(F, df1, df2))
    
    return {
        'F_statistic': F,
        'p_value': p_value,
        'var_x': var_x,
        'var_y': var_y,
        'var_ratio': var_x / var_y
    }


def levene_test(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Perform Levene's test for equality of variances.
    
    More robust to non-normality than F-test.
    Uses median-based version (Brown-Forsythe).
    
    Args:
        x: First sample array
        y: Second sample array
        
    Returns:
        Dictionary with test statistic and p-value
    """
    # Levene test with center='median' is the Brown-Forsythe test
    statistic, p_value = stats.levene(x, y, center='median')
    
    return {
        'statistic': statistic,
        'p_value': p_value
    }


def brown_forsythe_test(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Perform Brown-Forsythe test for equality of variances.
    
    This is Levene's test using median instead of mean.
    More robust to heavy-tailed distributions.
    
    Args:
        x: First sample array
        y: Second sample array
        
    Returns:
        Dictionary with test statistic and p-value
    """
    # Brown-Forsythe is Levene with center='median'
    return levene_test(x, y)


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    statistic_fn,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    random_seed: int = 42
) -> Dict[str, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Args:
        x: First sample array
        y: Second sample array
        statistic_fn: Function that takes (x, y) and returns a scalar statistic
        n_bootstrap: Number of bootstrap resamples
        alpha: Significance level (default 0.05 for 95% CI)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with point estimate, lower and upper CI bounds
    """
    np.random.seed(random_seed)
    
    n_x = len(x)
    n_y = len(y)
    
    # Compute point estimate on original data
    point_estimate = statistic_fn(x, y)
    
    # Bootstrap resampling
    bootstrap_stats = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Resample with replacement
        x_boot = np.random.choice(x, size=n_x, replace=True)
        y_boot = np.random.choice(y, size=n_y, replace=True)
        
        bootstrap_stats[i] = statistic_fn(x_boot, y_boot)
    
    # Percentile method for CI
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return {
        'point_estimate': point_estimate,
        'lower_ci': lower,
        'upper_ci': upper,
        'bootstrap_std': np.std(bootstrap_stats)
    }


def mean_difference(x: np.ndarray, y: np.ndarray) -> float:
    """Compute difference in means: mean(x) - mean(y)."""
    return np.mean(x) - np.mean(y)


def std_ratio(x: np.ndarray, y: np.ndarray) -> float:
    """Compute ratio of standard deviations: std(x) / std(y)."""
    return np.std(x, ddof=1) / np.std(y, ddof=1)


def variance_ratio(x: np.ndarray, y: np.ndarray) -> float:
    """Compute ratio of variances: var(x) / var(y)."""
    return np.var(x, ddof=1) / np.var(y, ddof=1)


class StrategyComparison:
    """
    Comprehensive statistical comparison of two strategies.
    
    Computes summary statistics, hypothesis tests, and bootstrap CIs.
    """
    
    def __init__(
        self,
        inv_data: np.ndarray,
        sym_data: np.ndarray,
        data_name: str,
        n_bootstrap: int = 10000,
        random_seed: int = 42
    ):
        """
        Initialize comparison.
        
        Args:
            inv_data: Data from inventory strategy
            sym_data: Data from symmetric strategy
            data_name: Name of the data (e.g., "Profit", "Inventory")
            n_bootstrap: Number of bootstrap resamples
            random_seed: Random seed for reproducibility
        """
        self.inv_data = inv_data
        self.sym_data = sym_data
        self.data_name = data_name
        self.n_bootstrap = n_bootstrap
        self.random_seed = random_seed
        
        # Compute all statistics
        self._compute_statistics()
    
    def _compute_statistics(self):
        """Compute all summary statistics and tests."""
        # Summary statistics
        self.mean_inv = np.mean(self.inv_data)
        self.mean_sym = np.mean(self.sym_data)
        self.std_inv = np.std(self.inv_data, ddof=1)
        self.std_sym = np.std(self.sym_data, ddof=1)
        
        # Standard errors
        self.se_inv = self.std_inv / np.sqrt(len(self.inv_data))
        self.se_sym = self.std_sym / np.sqrt(len(self.sym_data))
        
        # Variance tests
        self.f_test_result = f_test_variance_equality(self.inv_data, self.sym_data)
        self.levene_result = levene_test(self.inv_data, self.sym_data)
        self.brown_forsythe_result = brown_forsythe_test(self.inv_data, self.sym_data)
        
        # Bootstrap CIs
        self.mean_diff_ci = bootstrap_ci(
            self.inv_data, self.sym_data, mean_difference,
            self.n_bootstrap, random_seed=self.random_seed
        )
        
        self.std_ratio_ci = bootstrap_ci(
            self.inv_data, self.sym_data, std_ratio,
            self.n_bootstrap, random_seed=self.random_seed
        )
        
        self.var_ratio_ci = bootstrap_ci(
            self.inv_data, self.sym_data, variance_ratio,
            self.n_bootstrap, random_seed=self.random_seed
        )
    
    def print_summary(self):
        """Print formatted summary of comparison."""
        print(f"\n{'='*70}")
        print(f"Statistical Comparison: {self.data_name}")
        print(f"{'='*70}")
        
        print(f"\nSummary Statistics:")
        print(f"  Inventory Strategy:")
        print(f"    Mean: {self.mean_inv:.4f} (SE: {self.se_inv:.4f})")
        print(f"    Std:  {self.std_inv:.4f}")
        print(f"  Symmetric Strategy:")
        print(f"    Mean: {self.mean_sym:.4f} (SE: {self.se_sym:.4f})")
        print(f"    Std:  {self.std_sym:.4f}")
        
        print(f"\nVariance Equality Tests:")
        print(f"  F-test:")
        print(f"    F-statistic: {self.f_test_result['F_statistic']:.4f}")
        print(f"    p-value: {self.f_test_result['p_value']:.4e}")
        print(f"    Variance ratio (inv/sym): {self.f_test_result['var_ratio']:.4f}")
        
        print(f"  Levene test (median):")
        print(f"    Statistic: {self.levene_result['statistic']:.4f}")
        print(f"    p-value: {self.levene_result['p_value']:.4e}")
        
        print(f"  Brown-Forsythe test:")
        print(f"    Statistic: {self.brown_forsythe_result['statistic']:.4f}")
        print(f"    p-value: {self.brown_forsythe_result['p_value']:.4e}")
        
        print(f"\nBootstrap Confidence Intervals (95%, {self.n_bootstrap} resamples):")
        print(f"  Mean difference (inv - sym):")
        print(f"    Point estimate: {self.mean_diff_ci['point_estimate']:.4f}")
        print(f"    95% CI: [{self.mean_diff_ci['lower_ci']:.4f}, {self.mean_diff_ci['upper_ci']:.4f}]")
        
        print(f"  Std ratio (inv / sym):")
        print(f"    Point estimate: {self.std_ratio_ci['point_estimate']:.4f}")
        print(f"    95% CI: [{self.std_ratio_ci['lower_ci']:.4f}, {self.std_ratio_ci['upper_ci']:.4f}]")
        
        print(f"  Variance ratio (inv / sym):")
        print(f"    Point estimate: {self.var_ratio_ci['point_estimate']:.4f}")
        print(f"    95% CI: [{self.var_ratio_ci['lower_ci']:.4f}, {self.var_ratio_ci['upper_ci']:.4f}]")
    
    def get_summary_dict(self) -> Dict:
        """
        Get summary as dictionary for programmatic access.
        
        Returns:
            Dictionary with all statistics and test results
        """
        return {
            'data_name': self.data_name,
            'mean_inv': self.mean_inv,
            'mean_sym': self.mean_sym,
            'std_inv': self.std_inv,
            'std_sym': self.std_sym,
            'se_inv': self.se_inv,
            'se_sym': self.se_sym,
            'f_test': self.f_test_result,
            'levene_test': self.levene_result,
            'brown_forsythe_test': self.brown_forsythe_result,
            'mean_diff_ci': self.mean_diff_ci,
            'std_ratio_ci': self.std_ratio_ci,
            'var_ratio_ci': self.var_ratio_ci
        }
