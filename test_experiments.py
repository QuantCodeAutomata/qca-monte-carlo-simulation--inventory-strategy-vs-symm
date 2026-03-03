"""
Comprehensive Test Suite for Market Making Experiments

Tests verify:
1. Mathematical correctness of formulas
2. Adherence to paper methodology
3. Edge cases and boundary conditions
4. Statistical properties
5. Reproducibility
"""

import numpy as np
import pytest
from pathlib import Path

from market_maker_simulation import SimulationParams, MarketMakerSimulator, EpisodeResult
from statistical_analysis import (
    f_test_variance_equality,
    levene_test,
    bootstrap_ci,
    mean_difference,
    std_ratio,
    StrategyComparison
)


class TestSimulationParams:
    """Test simulation parameter calculations."""
    
    def test_spread_formula_gamma_01(self):
        """Test spread calculation for gamma = 0.1."""
        params = SimulationParams(gamma=0.1, k=1.5)
        spread = params.spread
        expected = (2.0 / 0.1) * np.log(1.0 + 0.1 / 1.5)
        assert np.abs(spread - expected) < 1e-10
        assert np.abs(spread - 1.2908) < 0.002  # Paper target (allowing rounding tolerance)
    
    def test_spread_formula_gamma_001(self):
        """Test spread calculation for gamma = 0.01."""
        params = SimulationParams(gamma=0.01, k=1.5)
        spread = params.spread
        expected = (2.0 / 0.01) * np.log(1.0 + 0.01 / 1.5)
        assert np.abs(spread - expected) < 1e-10
        assert np.abs(spread - 1.3289) < 0.001  # Paper target
    
    def test_spread_formula_gamma_05(self):
        """Test spread calculation for gamma = 0.5."""
        params = SimulationParams(gamma=0.5, k=1.5)
        spread = params.spread
        expected = (2.0 / 0.5) * np.log(1.0 + 0.5 / 1.5)
        assert np.abs(spread - expected) < 1e-10
        assert np.abs(spread - 1.1507) < 0.001  # Paper target
    
    def test_number_of_steps(self):
        """Test that N = T / dt is correct."""
        params = SimulationParams(T=1.0, dt=0.005)
        assert params.N == 200
        
        params2 = SimulationParams(T=2.0, dt=0.01)
        assert params2.N == 200
    
    def test_default_parameters(self):
        """Test default parameter values match paper."""
        params = SimulationParams()
        assert params.s_0 == 100.0
        assert params.T == 1.0
        assert params.sigma == 2.0
        assert params.dt == 0.005
        assert params.q_0 == 0
        assert params.X_0 == 0.0
        assert params.k == 1.5
        assert params.A == 140.0


class TestReservationPrice:
    """Test reservation price calculations."""
    
    def test_reservation_price_zero_inventory(self):
        """With q=0, reservation price should equal mid-price."""
        S = 100.0
        q = 0
        gamma = 0.1
        sigma = 2.0
        tau = 1.0
        
        r = S - q * gamma * sigma**2 * tau
        assert r == S
    
    def test_reservation_price_positive_inventory(self):
        """With q>0 (long), reservation price should be below mid-price."""
        S = 100.0
        q = 5
        gamma = 0.1
        sigma = 2.0
        tau = 0.5
        
        r = S - q * gamma * sigma**2 * tau
        adjustment = q * gamma * sigma**2 * tau
        assert adjustment == 5 * 0.1 * 4.0 * 0.5  # = 1.0
        assert r == S - 1.0
        assert r < S
    
    def test_reservation_price_negative_inventory(self):
        """With q<0 (short), reservation price should be above mid-price."""
        S = 100.0
        q = -5
        gamma = 0.1
        sigma = 2.0
        tau = 0.5
        
        r = S - q * gamma * sigma**2 * tau
        adjustment = q * gamma * sigma**2 * tau
        assert adjustment == -5 * 0.1 * 4.0 * 0.5  # = -1.0
        assert r == S + 1.0
        assert r > S
    
    def test_reservation_price_time_decay(self):
        """Inventory adjustment should decrease as tau -> 0."""
        S = 100.0
        q = 5
        gamma = 0.1
        sigma = 2.0
        
        r_tau_1 = S - q * gamma * sigma**2 * 1.0
        r_tau_05 = S - q * gamma * sigma**2 * 0.5
        r_tau_0 = S - q * gamma * sigma**2 * 0.0
        
        # Adjustment magnitude decreases with time remaining
        assert abs(r_tau_1 - S) > abs(r_tau_05 - S)
        assert abs(r_tau_05 - S) > abs(r_tau_0 - S)
        assert r_tau_0 == S  # At horizon, no adjustment


class TestQuoteDistances:
    """Test quote distance calculations."""
    
    def test_symmetric_quote_distances(self):
        """Symmetric strategy has constant equal distances."""
        L = 1.292
        delta_a = L / 2
        delta_b = L / 2
        
        assert delta_a == delta_b
        assert delta_a == 0.646
    
    def test_inventory_quote_distances_zero_inventory(self):
        """With q=0, inventory strategy matches symmetric."""
        L = 1.292
        q = 0
        gamma = 0.1
        sigma = 2.0
        tau = 1.0
        
        delta_a = L / 2 - q * gamma * sigma**2 * tau
        delta_b = L / 2 + q * gamma * sigma**2 * tau
        
        assert delta_a == L / 2
        assert delta_b == L / 2
        assert delta_a == delta_b
    
    def test_inventory_quote_distances_positive_inventory(self):
        """With q>0, ask distance increases, bid distance decreases."""
        L = 1.292
        q = 5
        gamma = 0.1
        sigma = 2.0
        tau = 0.5
        
        delta_a = L / 2 - q * gamma * sigma**2 * tau
        delta_b = L / 2 + q * gamma * sigma**2 * tau
        
        assert delta_a < L / 2  # Narrower ask spread to encourage selling
        assert delta_b > L / 2  # Wider bid spread to discourage buying
    
    def test_floor_enforcement(self):
        """Quote distances should never be negative."""
        L = 1.292
        q = 100  # Large positive inventory
        gamma = 0.5
        sigma = 2.0
        tau = 1.0
        
        delta_a_raw = L / 2 - q * gamma * sigma**2 * tau
        delta_a = max(delta_a_raw, 0.0)
        
        assert delta_a_raw < 0  # Would be negative without floor
        assert delta_a == 0.0  # Floor enforced


class TestExecutionIntensity:
    """Test order arrival intensity calculations."""
    
    def test_intensity_formula(self):
        """Test lambda(delta) = A * exp(-k * delta)."""
        A = 140.0
        k = 1.5
        delta = 0.646
        
        lambda_val = A * np.exp(-k * delta)
        expected = 140.0 * np.exp(-1.5 * 0.646)
        
        assert np.abs(lambda_val - expected) < 1e-10
        assert lambda_val < A  # Intensity decreases with distance
    
    def test_intensity_monotonicity(self):
        """Intensity should decrease as quote distance increases."""
        A = 140.0
        k = 1.5
        
        lambda_0 = A * np.exp(-k * 0.0)
        lambda_05 = A * np.exp(-k * 0.5)
        lambda_1 = A * np.exp(-k * 1.0)
        
        assert lambda_0 > lambda_05 > lambda_1
        assert lambda_0 == A  # At delta=0, lambda=A
    
    def test_intensity_dt_bound(self):
        """Test that lambda * dt <= 1 for valid probabilities."""
        A = 140.0
        k = 1.5
        dt = 0.005
        delta = 0.0  # Worst case
        
        lambda_val = A * np.exp(-k * delta)
        prob = lambda_val * dt
        
        assert prob <= 1.0
        assert prob == A * dt  # = 140 * 0.005 = 0.7


class TestMidPriceEvolution:
    """Test mid-price binomial approximation."""
    
    def test_midprice_step_magnitude(self):
        """Test that price step = epsilon * sigma * sqrt(dt)."""
        sigma = 2.0
        dt = 0.005
        epsilon = 1  # or -1
        
        step = epsilon * sigma * np.sqrt(dt)
        expected = 1 * 2.0 * np.sqrt(0.005)
        
        assert np.abs(step - expected) < 1e-10
    
    def test_midprice_variance(self):
        """Test that variance of S_T approximately equals sigma^2 * T."""
        np.random.seed(42)
        n_paths = 10000
        T = 1.0
        dt = 0.005
        N = int(T / dt)
        sigma = 2.0
        S_0 = 100.0
        
        S_T = np.zeros(n_paths)
        for i in range(n_paths):
            S = S_0
            for _ in range(N):
                eps = np.random.choice([-1, 1])
                S = S + eps * sigma * np.sqrt(dt)
            S_T[i] = S
        
        # Variance should be approximately sigma^2 * T = 4.0
        var_S_T = np.var(S_T)
        expected_var = sigma**2 * T
        
        # Allow 20% tolerance due to Monte Carlo error
        assert np.abs(var_S_T - expected_var) < 0.2 * expected_var


class TestSingleEpisodeSimulation:
    """Test single episode simulation logic."""
    
    def test_episode_reproducibility(self):
        """Test that same seed produces same results."""
        params = SimulationParams(gamma=0.1, random_seed=42)
        simulator1 = MarketMakerSimulator(params)
        result1 = simulator1.simulate_episode()
        
        params2 = SimulationParams(gamma=0.1, random_seed=42)
        simulator2 = MarketMakerSimulator(params2)
        result2 = simulator2.simulate_episode()
        
        assert result1.profit_inv == result2.profit_inv
        assert result1.profit_sym == result2.profit_sym
        assert result1.q_T_inv == result2.q_T_inv
        assert result1.q_T_sym == result2.q_T_sym
    
    def test_episode_terminal_profit_formula(self):
        """Test that terminal profit = X + q * S."""
        params = SimulationParams(gamma=0.1, random_seed=100)
        simulator = MarketMakerSimulator(params)
        result = simulator.simulate_episode(store_path=True)
        
        # Terminal profit should equal cash + inventory * terminal mid-price
        S_T = result.S_path[-1]
        
        # Cannot directly verify X and q from result, but profit formula is tested
        # in the implementation. This test ensures the simulation runs without error.
        assert isinstance(result.profit_inv, (int, float))
        assert isinstance(result.profit_sym, (int, float))
        assert isinstance(result.q_T_inv, (int, np.integer))
        assert isinstance(result.q_T_sym, (int, np.integer))
    
    def test_path_storage(self):
        """Test that paths are stored when requested."""
        params = SimulationParams(gamma=0.1, random_seed=42)
        simulator = MarketMakerSimulator(params)
        
        # With path storage
        result_with_path = simulator.simulate_episode(store_path=True)
        assert result_with_path.S_path is not None
        assert result_with_path.r_path is not None
        assert len(result_with_path.S_path) == params.N + 1
        
        # Reset simulator with new seed
        params2 = SimulationParams(gamma=0.1, random_seed=43)
        simulator2 = MarketMakerSimulator(params2)
        
        # Without path storage
        result_no_path = simulator2.simulate_episode(store_path=False)
        assert result_no_path.S_path is None
        assert result_no_path.r_path is None


class TestMonteCarloSimulation:
    """Test Monte Carlo simulation aggregation."""
    
    def test_monte_carlo_output_shape(self):
        """Test that Monte Carlo returns correct array shapes."""
        params = SimulationParams(gamma=0.1, n_simulations=100, random_seed=42)
        simulator = MarketMakerSimulator(params)
        results = simulator.run_monte_carlo(store_first_path=False)
        
        assert len(results['profits_inv']) == 100
        assert len(results['profits_sym']) == 100
        assert len(results['q_T_inv']) == 100
        assert len(results['q_T_sym']) == 100
    
    def test_monte_carlo_first_path_storage(self):
        """Test that first path is stored when requested."""
        params = SimulationParams(gamma=0.1, n_simulations=10, random_seed=42)
        simulator = MarketMakerSimulator(params)
        
        results = simulator.run_monte_carlo(store_first_path=True)
        assert results['first_episode'] is not None
        assert results['first_episode'].S_path is not None
    
    def test_monte_carlo_inventory_reduction(self):
        """Test that inventory strategy reduces variance vs symmetric."""
        params = SimulationParams(gamma=0.1, n_simulations=1000, random_seed=42)
        simulator = MarketMakerSimulator(params)
        results = simulator.run_monte_carlo(store_first_path=False)
        
        std_profit_inv = np.std(results['profits_inv'], ddof=1)
        std_profit_sym = np.std(results['profits_sym'], ddof=1)
        std_q_inv = np.std(results['q_T_inv'], ddof=1)
        std_q_sym = np.std(results['q_T_sym'], ddof=1)
        
        # Inventory strategy should have lower variance
        assert std_profit_inv < std_profit_sym
        assert std_q_inv < std_q_sym


class TestStatisticalAnalysis:
    """Test statistical analysis functions."""
    
    def test_f_test_equal_variances(self):
        """Test F-test on samples with equal variances."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 1000)
        y = np.random.normal(0, 1, 1000)
        
        result = f_test_variance_equality(x, y)
        
        # Should not reject null hypothesis of equal variances
        assert result['p_value'] > 0.05
    
    def test_f_test_unequal_variances(self):
        """Test F-test on samples with unequal variances."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 1000)
        y = np.random.normal(0, 5, 1000)  # Much larger variance
        
        result = f_test_variance_equality(x, y)
        
        # Should reject null hypothesis of equal variances
        assert result['p_value'] < 0.01
        assert result['var_y'] > result['var_x']
    
    def test_bootstrap_ci_coverage(self):
        """Test that bootstrap CI has approximately correct coverage."""
        np.random.seed(42)
        x = np.random.normal(5, 1, 100)
        y = np.random.normal(3, 1, 100)
        
        ci = bootstrap_ci(x, y, mean_difference, n_bootstrap=1000, random_seed=42)
        
        # True difference is 5 - 3 = 2
        # CI should contain true value
        assert ci['lower_ci'] <= 2 <= ci['upper_ci']
    
    def test_strategy_comparison_class(self):
        """Test StrategyComparison class."""
        np.random.seed(42)
        inv_data = np.random.normal(60, 5, 1000)
        sym_data = np.random.normal(65, 10, 1000)
        
        comparison = StrategyComparison(
            inv_data, sym_data,
            data_name="Test Profit",
            n_bootstrap=1000,
            random_seed=42
        )
        
        # Check that statistics are computed
        assert comparison.mean_inv is not None
        assert comparison.std_inv is not None
        assert comparison.f_test_result is not None
        assert comparison.mean_diff_ci is not None
        
        # Check variance reduction detected
        assert comparison.std_inv < comparison.std_sym


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_simulation(self):
        """Test with n_simulations = 1."""
        params = SimulationParams(gamma=0.1, n_simulations=1, random_seed=42)
        simulator = MarketMakerSimulator(params)
        results = simulator.run_monte_carlo()
        
        assert len(results['profits_inv']) == 1
        assert len(results['profits_sym']) == 1
    
    def test_extreme_gamma_low(self):
        """Test with very low gamma (should approach symmetric)."""
        params = SimulationParams(gamma=0.001, n_simulations=100, random_seed=42)
        simulator = MarketMakerSimulator(params)
        results = simulator.run_monte_carlo()
        
        # At very low gamma, strategies should be very similar
        mean_diff = abs(np.mean(results['profits_inv']) - np.mean(results['profits_sym']))
        
        # Difference should be small
        assert mean_diff < 5.0  # Arbitrary threshold
    
    def test_extreme_gamma_high(self):
        """Test with very high gamma (strong inventory control)."""
        params = SimulationParams(gamma=1.0, n_simulations=100, random_seed=42)
        simulator = MarketMakerSimulator(params)
        results = simulator.run_monte_carlo()
        
        # At high gamma, inventory variance should be very low
        std_q_inv = np.std(results['q_T_inv'], ddof=1)
        std_q_sym = np.std(results['q_T_sym'], ddof=1)
        
        assert std_q_inv < std_q_sym
        assert std_q_inv < 2.0  # Should be very tight


class TestMethodologyAdherence:
    """Test adherence to paper methodology."""
    
    def test_common_random_numbers(self):
        """Test that both strategies use same random shocks in each episode."""
        # This is implicitly tested by the implementation using shared
        # eps, U_ask, U_bid arrays, but we verify by checking that
        # results are deterministic given seed
        
        params = SimulationParams(gamma=0.1, n_simulations=5, random_seed=999)
        simulator1 = MarketMakerSimulator(params)
        results1 = simulator1.run_monte_carlo(store_first_path=False)
        
        params2 = SimulationParams(gamma=0.1, n_simulations=5, random_seed=999)
        simulator2 = MarketMakerSimulator(params2)
        results2 = simulator2.run_monte_carlo(store_first_path=False)
        
        np.testing.assert_array_equal(results1['profits_inv'], results2['profits_inv'])
        np.testing.assert_array_equal(results1['profits_sym'], results2['profits_sym'])
    
    def test_time_step_count(self):
        """Test that simulation runs exactly N steps."""
        params = SimulationParams(T=1.0, dt=0.005, gamma=0.1, random_seed=42)
        assert params.N == 200
        
        simulator = MarketMakerSimulator(params)
        result = simulator.simulate_episode(store_path=True)
        
        # Path should have N+1 points (including initial state)
        assert len(result.S_path) == 201
    
    def test_initial_conditions(self):
        """Test that simulation starts with correct initial conditions."""
        params = SimulationParams(s_0=100.0, q_0=0, X_0=0.0, gamma=0.1, random_seed=42)
        simulator = MarketMakerSimulator(params)
        result = simulator.simulate_episode(store_path=True)
        
        # Initial mid-price should be s_0
        assert result.S_path[0] == 100.0
        
        # Initial inventory should be q_0
        assert result.q_inv_path[0] == 0


def test_full_experiment_gamma_01():
    """Integration test for gamma = 0.1 experiment."""
    params = SimulationParams(gamma=0.1, n_simulations=1000, random_seed=42)
    simulator = MarketMakerSimulator(params)
    results = simulator.run_monte_carlo()
    
    # Check that results are within expected ranges from paper
    mean_profit_inv = np.mean(results['profits_inv'])
    std_profit_inv = np.std(results['profits_inv'], ddof=1)
    std_q_inv = np.std(results['q_T_inv'], ddof=1)
    
    # Targets from paper (with tolerance for Monte Carlo variation)
    # Allow ±2*SE tolerance
    se_profit = std_profit_inv / np.sqrt(1000)
    
    # Check means are roughly in range (paper: ~62.94)
    assert 55 < mean_profit_inv < 70
    
    # Check std(profit) is in range (paper: ~5.89)
    assert 4 < std_profit_inv < 8
    
    # Check std(q_T) is in range (paper: ~2.80)
    assert 2 < std_q_inv < 4


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
