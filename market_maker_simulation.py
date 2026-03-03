"""
Market Maker Simulation: Inventory-based vs Symmetric Quoting Strategies

This module implements Monte Carlo simulation of market making strategies
as described in the research paper on optimal market making with inventory risk.

The simulation compares:
1. Inventory strategy: quotes centered around reservation price r_t = S_t - q_t * gamma * sigma^2 * tau
2. Symmetric strategy: quotes centered around mid-price S_t (no inventory adjustment)

Key features:
- Binomial approximation to Brownian motion for mid-price evolution
- Exponential order arrival intensity: lambda(delta) = A * exp(-k * delta)
- Common random numbers for paired comparison
- Proper handling of quote floor constraints
"""

import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class SimulationParams:
    """Parameters for market making simulation."""
    s_0: float = 100.0          # Initial mid-price
    T: float = 1.0              # Time horizon
    sigma: float = 2.0          # Mid-price volatility
    dt: float = 0.005           # Time step
    q_0: int = 0                # Initial inventory
    X_0: float = 0.0            # Initial cash position
    gamma: float = 0.1          # Risk aversion parameter
    k: float = 1.5              # Order arrival intensity decay
    A: float = 140.0            # Order arrival intensity scale
    n_simulations: int = 1000   # Number of Monte Carlo episodes
    random_seed: int = 42       # Random seed for reproducibility
    
    @property
    def N(self) -> int:
        """Number of time steps."""
        return int(self.T / self.dt)
    
    @property
    def spread(self) -> float:
        """Optimal constant spread L(gamma) = (2/gamma) * ln(1 + gamma/k)."""
        return (2.0 / self.gamma) * np.log(1.0 + self.gamma / self.k)


@dataclass
class EpisodeResult:
    """Results from a single simulation episode."""
    profit_inv: float           # Terminal profit for inventory strategy
    profit_sym: float           # Terminal profit for symmetric strategy
    q_T_inv: int                # Terminal inventory for inventory strategy
    q_T_sym: int                # Terminal inventory for symmetric strategy
    S_path: np.ndarray = None   # Mid-price path (optional, for plotting)
    r_path: np.ndarray = None   # Reservation price path (optional, for plotting)
    p_ask_path: np.ndarray = None  # Ask price path (optional, for plotting)
    p_bid_path: np.ndarray = None  # Bid price path (optional, for plotting)
    q_inv_path: np.ndarray = None  # Inventory path (optional, for plotting)


class MarketMakerSimulator:
    """
    Monte Carlo simulator for market making strategies.
    
    Simulates both inventory-based and symmetric quoting strategies
    using common random numbers for paired comparison.
    """
    
    def __init__(self, params: SimulationParams):
        """
        Initialize simulator with given parameters.
        
        Args:
            params: Simulation parameters including gamma, sigma, A, k, etc.
        """
        self.params = params
        np.random.seed(params.random_seed)
        
        # Verify spread calculation
        spread = self.params.spread
        print(f"Risk aversion gamma: {params.gamma}")
        print(f"Computed spread L(gamma): {spread:.4f}")
        print(f"Number of time steps N: {params.N}")
        print(f"Random seed: {params.random_seed}")
    
    def simulate_episode(self, store_path: bool = False) -> EpisodeResult:
        """
        Simulate a single episode for both strategies using common random numbers.
        
        Args:
            store_path: If True, store full price and inventory paths for plotting
            
        Returns:
            EpisodeResult containing terminal profits, inventories, and optional paths
        """
        params = self.params
        N = params.N
        dt = params.dt
        sigma = params.sigma
        gamma = params.gamma
        A = params.A
        k = params.k
        L = params.spread
        
        # Pre-generate common random numbers for paired comparison
        # Mid-price shocks: +1 or -1 with equal probability
        eps = np.random.choice([-1, 1], size=N)
        # Execution events: uniform [0,1] for ask and bid independently
        U_ask = np.random.uniform(0, 1, size=N)
        U_bid = np.random.uniform(0, 1, size=N)
        
        # Initialize state variables
        S = params.s_0  # Current mid-price
        
        # Inventory strategy state
        X_inv = params.X_0
        q_inv = params.q_0
        
        # Symmetric strategy state
        X_sym = params.X_0
        q_sym = params.q_0
        
        # Optional path storage for plotting
        if store_path:
            S_path = np.zeros(N + 1)
            r_path = np.zeros(N + 1)
            p_ask_path = np.zeros(N + 1)
            p_bid_path = np.zeros(N + 1)
            q_inv_path = np.zeros(N + 1)
            
            S_path[0] = S
            r_path[0] = S - q_inv * gamma * sigma**2 * params.T
            p_ask_path[0] = r_path[0] + L / 2
            p_bid_path[0] = r_path[0] - L / 2
            q_inv_path[0] = q_inv
        
        # Simulation loop over time steps
        for n in range(N):
            # Time remaining until horizon
            tau = params.T - n * dt
            
            # === INVENTORY STRATEGY ===
            # Reservation price: r = S - q * gamma * sigma^2 * tau
            r = S - q_inv * gamma * sigma**2 * tau
            
            # Quote distances from mid-price
            # Ask distance: delta^a = (r + L/2) - S = L/2 - q * gamma * sigma^2 * tau
            # Bid distance: delta^b = S - (r - L/2) = L/2 + q * gamma * sigma^2 * tau
            delta_a_inv = L / 2 - q_inv * gamma * sigma**2 * tau
            delta_b_inv = L / 2 + q_inv * gamma * sigma**2 * tau
            
            # Enforce floor to prevent quote crossing and lambda*dt > 1
            delta_a_inv = max(delta_a_inv, 0.0)
            delta_b_inv = max(delta_b_inv, 0.0)
            
            # Quote prices
            p_ask_inv = S + delta_a_inv
            p_bid_inv = S - delta_b_inv
            
            # === SYMMETRIC STRATEGY ===
            # Constant spread around mid-price
            delta_a_sym = L / 2
            delta_b_sym = L / 2
            
            p_ask_sym = S + delta_a_sym
            p_bid_sym = S - delta_b_sym
            
            # === COMPUTE EXECUTION PROBABILITIES ===
            # Intensity: lambda(delta) = A * exp(-k * delta)
            lambda_a_inv = A * np.exp(-k * delta_a_inv)
            lambda_b_inv = A * np.exp(-k * delta_b_inv)
            
            lambda_a_sym = A * np.exp(-k * delta_a_sym)
            lambda_b_sym = A * np.exp(-k * delta_b_sym)
            
            # Execution probability = lambda * dt (Bernoulli approximation)
            prob_ask_inv = lambda_a_inv * dt
            prob_bid_inv = lambda_b_inv * dt
            
            prob_ask_sym = lambda_a_sym * dt
            prob_bid_sym = lambda_b_sym * dt
            
            # Sanity check: probabilities should be <= 1
            assert prob_ask_inv <= 1.0, f"prob_ask_inv = {prob_ask_inv} > 1"
            assert prob_bid_inv <= 1.0, f"prob_bid_inv = {prob_bid_inv} > 1"
            assert prob_ask_sym <= 1.0, f"prob_ask_sym = {prob_ask_sym} > 1"
            assert prob_bid_sym <= 1.0, f"prob_bid_sym = {prob_bid_sym} > 1"
            
            # === SIMULATE EXECUTIONS (using shared random numbers) ===
            # Inventory strategy
            if U_ask[n] < prob_ask_inv:  # Ask executed: sell one unit
                q_inv -= 1
                X_inv += p_ask_inv  # Receive ask price
            
            if U_bid[n] < prob_bid_inv:  # Bid executed: buy one unit
                q_inv += 1
                X_inv -= p_bid_inv  # Pay bid price
            
            # Symmetric strategy
            if U_ask[n] < prob_ask_sym:  # Ask executed: sell one unit
                q_sym -= 1
                X_sym += p_ask_sym  # Receive ask price
            
            if U_bid[n] < prob_bid_sym:  # Bid executed: buy one unit
                q_sym += 1
                X_sym -= p_bid_sym  # Pay bid price
            
            # === UPDATE MID-PRICE (binomial approximation to Brownian motion) ===
            # S_t+dt = S_t + eps * sigma * sqrt(dt), where eps in {-1, +1}
            S = S + eps[n] * sigma * np.sqrt(dt)
            
            # Store path if requested
            if store_path:
                S_path[n + 1] = S
                r_path[n + 1] = S - q_inv * gamma * sigma**2 * tau
                p_ask_path[n + 1] = r_path[n + 1] + L / 2
                p_bid_path[n + 1] = r_path[n + 1] - L / 2
                q_inv_path[n + 1] = q_inv
        
        # === COMPUTE TERMINAL PROFITS ===
        # Profit = Cash + Inventory * Terminal_Mid_Price
        profit_inv = X_inv + q_inv * S
        profit_sym = X_sym + q_sym * S
        
        # Return results
        if store_path:
            return EpisodeResult(
                profit_inv=profit_inv,
                profit_sym=profit_sym,
                q_T_inv=q_inv,
                q_T_sym=q_sym,
                S_path=S_path,
                r_path=r_path,
                p_ask_path=p_ask_path,
                p_bid_path=p_bid_path,
                q_inv_path=q_inv_path
            )
        else:
            return EpisodeResult(
                profit_inv=profit_inv,
                profit_sym=profit_sym,
                q_T_inv=q_inv,
                q_T_sym=q_sym
            )
    
    def run_monte_carlo(self, store_first_path: bool = True) -> Dict[str, np.ndarray]:
        """
        Run Monte Carlo simulation for n_simulations episodes.
        
        Args:
            store_first_path: If True, store full path for episode 0 for plotting
            
        Returns:
            Dictionary containing arrays of results:
                - profits_inv: Terminal profits for inventory strategy (n_simulations,)
                - profits_sym: Terminal profits for symmetric strategy (n_simulations,)
                - q_T_inv: Terminal inventories for inventory strategy (n_simulations,)
                - q_T_sym: Terminal inventories for symmetric strategy (n_simulations,)
                - first_episode: EpisodeResult with full paths (if store_first_path=True)
        """
        n_sims = self.params.n_simulations
        
        # Pre-allocate result arrays
        profits_inv = np.zeros(n_sims)
        profits_sym = np.zeros(n_sims)
        q_T_inv = np.zeros(n_sims, dtype=int)
        q_T_sym = np.zeros(n_sims, dtype=int)
        
        first_episode = None
        
        print(f"\nRunning {n_sims} Monte Carlo simulations...")
        
        for i in range(n_sims):
            if (i + 1) % 100 == 0:
                print(f"  Completed {i + 1}/{n_sims} episodes")
            
            # Store path for first episode for plotting
            store_path = (i == 0 and store_first_path)
            result = self.simulate_episode(store_path=store_path)
            
            profits_inv[i] = result.profit_inv
            profits_sym[i] = result.profit_sym
            q_T_inv[i] = result.q_T_inv
            q_T_sym[i] = result.q_T_sym
            
            if i == 0 and store_first_path:
                first_episode = result
        
        print(f"Completed all {n_sims} episodes.\n")
        
        return {
            'profits_inv': profits_inv,
            'profits_sym': profits_sym,
            'q_T_inv': q_T_inv,
            'q_T_sym': q_T_sym,
            'first_episode': first_episode
        }
