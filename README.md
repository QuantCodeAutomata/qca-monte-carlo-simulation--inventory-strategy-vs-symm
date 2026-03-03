# Monte Carlo Simulation: Inventory Strategy vs Symmetric Strategy

This repository implements Monte Carlo simulations comparing two market-making quoting strategies:

1. **Inventory Strategy**: Quotes centered around the reservation (indifference) price, which adjusts based on current inventory position
2. **Symmetric Strategy**: Quotes centered around the mid-price with no inventory adjustment

## Overview

The experiments reproduce results from a research paper on optimal market making with inventory risk. The inventory-based strategy dynamically adjusts quote placement to manage inventory risk, resulting in:

- **Lower variance** of terminal profit and inventory
- **Tighter control** of inventory positions
- Trade-off with expected profit at high risk aversion levels

## Repository Structure

```
.
├── market_maker_simulation.py   # Core simulation engine
├── statistical_analysis.py      # Statistical tests and comparisons
├── visualization.py             # Plotting and table generation
├── run_experiments.py           # Main experiment runner
├── test_experiments.py          # Comprehensive test suite
├── results/                     # Output directory
│   ├── RESULTS.md              # Comprehensive results summary
│   ├── histogram_*.png         # Distribution plots
│   ├── path_plot_*.png         # Price path plots
│   └── raw_data_*.csv          # Raw simulation data
└── README.md                    # This file
```

## Experiments

### Experiment 1: Moderate Risk Aversion (γ = 0.1)

- **Objective**: Demonstrate variance reduction with minimal profit sacrifice
- **Key Finding**: Inventory strategy achieves ~2.3x reduction in profit variance and ~3.1x reduction in inventory variance

### Experiment 2: Low Risk Aversion (γ = 0.01)

- **Objective**: Test convergence behavior as γ → 0
- **Key Finding**: Strategies become more similar at low risk aversion, validating theoretical convergence

### Experiment 3: High Risk Aversion (γ = 0.5)

- **Objective**: Demonstrate strong variance reduction with profit trade-off
- **Key Finding**: Inventory strategy achieves ~3.1x reduction in profit variance and ~4.8x reduction in inventory variance, but sacrifices ~50% of expected profit

## Methodology

### Model Parameters

- **Mid-price evolution**: Binomial approximation to Brownian motion
  - Initial price: S₀ = 100
  - Volatility: σ = 2
  - Time horizon: T = 1
  - Time step: dt = 0.005 (200 steps)

- **Order arrival intensity**: λ(δ) = A · exp(-k · δ)
  - Scale parameter: A = 140
  - Decay parameter: k = 1.5

- **Optimal spread**: L(γ) = (2/γ) · ln(1 + γ/k)

### Reservation Price

The inventory strategy centers quotes around the reservation price:

```
r = S - q · γ · σ² · τ
```

Where:
- S = current mid-price
- q = current inventory position
- γ = risk aversion parameter
- σ = volatility
- τ = time remaining

### Quote Placement

**Inventory Strategy:**
- Ask: p^a = r + L/2
- Bid: p^b = r - L/2
- Distances from mid-price depend on inventory

**Symmetric Strategy:**
- Ask: p^a = S + L/2
- Bid: p^b = S - L/2
- Constant distances from mid-price

### Common Random Numbers

For fair comparison, both strategies use identical random shocks in each episode:
- Same mid-price evolution
- Same execution event draws

This enables paired statistical testing with higher power.

## Installation

### Prerequisites

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn pytest
```

### Clone and Run

```bash
git clone <repository-url>
cd qca-monte-carlo-simulation--inventory-strategy-vs-symm
python run_experiments.py
```

## Usage

### Run All Experiments

```bash
python run_experiments.py
```

This will:
1. Run 1000 Monte Carlo episodes for each of three γ values (0.01, 0.1, 0.5)
2. Perform statistical tests (F-test, Levene test, bootstrap CIs)
3. Generate histograms and path plots
4. Save results to `results/RESULTS.md`

### Run Tests

```bash
pytest test_experiments.py -v
```

Tests verify:
- Mathematical formula correctness
- Methodology adherence to paper specifications
- Edge cases and boundary conditions
- Statistical properties
- Reproducibility

## Results

Key findings across all experiments:

| γ    | Std(Profit) Ratio | Std(q_T) Ratio | Mean Profit Sacrifice |
|------|-------------------|----------------|-----------------------|
| 0.01 | ~0.65             | ~0.54          | Minimal (~0.6)        |
| 0.1  | ~0.44             | ~0.32          | Small (~4.3)          |
| 0.5  | ~0.32             | ~0.21          | Large (~32.3)         |

Ratio < 1 indicates inventory strategy has lower variance than symmetric strategy.

See `results/RESULTS.md` for comprehensive statistical analysis.

## Key Insights

1. **Variance Reduction**: The inventory strategy consistently reduces both profit and inventory variance across all risk aversion levels

2. **Profit-Variance Trade-off**: At high risk aversion (γ = 0.5), aggressive inventory management sacrifices ~50% of expected profit

3. **Convergence Property**: As γ → 0, the inventory strategy converges toward the symmetric strategy

4. **Statistical Significance**: All variance reductions are statistically significant (p < 0.001) based on F-tests and Levene tests

## Methodology Adherence

This implementation strictly follows the paper methodology:

✓ Exact parameter values from paper  
✓ Binomial approximation for mid-price evolution  
✓ Exponential order arrival intensity  
✓ Reservation price formula with inventory adjustment  
✓ Floor enforcement to prevent quote crossing  
✓ Common random numbers for paired comparison  
✓ 1000 Monte Carlo episodes per experiment  
✓ Bootstrap confidence intervals (10,000 resamples)  

## References

The experiments are based on optimal market making theory with exponential utility and Poisson order arrivals.

## License

MIT License

## Author

Generated by QCA Agent for quantitative finance research.
