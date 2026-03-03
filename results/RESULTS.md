# Monte Carlo Simulation Results

## Market Making: Inventory Strategy vs Symmetric Strategy

This document contains results from three Monte Carlo experiments comparing inventory-based and symmetric quoting strategies for market making.

---

## Experiment 1: Moderate Risk Aversion (γ = 0.1)

**Experiment ID:** exp_1  
**Risk Aversion (γ):** 0.1  
**Optimal Spread L(γ):** 1.2908  
**Number of Episodes:** 1000

### Summary Statistics

| Strategy | Mean(Profit) | Std(Profit) | Mean(q_T) | Std(q_T) |
|----------|--------------|-------------|-----------|----------|
| **Inventory** | 64.3573 (SE: 0.1832) | 5.7927 | 0.0200 (SE: 0.0951) | 3.0058 |
| **Symmetric** | 68.7123 (SE: 0.4238) | 13.4024 | 0.1590 (SE: 0.2788) | 8.8161 |

### Key Findings

- **Std(Profit) Reduction:** Inventory strategy has 0.4322x the variance of symmetric strategy
- **Std(q_T) Reduction:** Inventory strategy has 0.3409x the inventory variance of symmetric strategy
- **Mean Profit Difference:** -4.3550 (inventory - symmetric)

### Statistical Tests


#### Profit Variance Equality Tests

**F-test:**
- F-statistic: 5.3532
- p-value: 2.2204e-16
- Variance ratio (inv/sym): 0.1868

**Levene Test (Brown-Forsythe):**
- Statistic: 287.0890
- p-value: 2.8262e-60

**Bootstrap Confidence Intervals (95%, 10,000 resamples):**
- Mean difference (inv - sym): [-5.2652, -3.4527]
- Std ratio (inv / sym): [0.3961, 0.4715]
- Variance ratio (inv / sym): [0.1569, 0.2223]

#### Terminal Inventory (q_T) Variance Equality Tests

**F-test:**
- F-statistic: 8.6028
- p-value: 2.2204e-16
- Variance ratio (inv/sym): 0.1162

**Levene Test (Brown-Forsythe):**
- Statistic: 639.9150
- p-value: 1.0324e-122

**Bootstrap Confidence Intervals (95%, 10,000 resamples):**
- Mean difference (inv - sym): [-0.7140, 0.4490]
- Std ratio (inv / sym): [0.3205, 0.3623]
- Variance ratio (inv / sym): [0.1027, 0.1313]

---

## Experiment 2: Low Risk Aversion (γ = 0.01) - Convergence Test

**Experiment ID:** exp_2  
**Risk Aversion (γ):** 0.01  
**Optimal Spread L(γ):** 1.3289  
**Number of Episodes:** 1000

### Summary Statistics

| Strategy | Mean(Profit) | Std(Profit) | Mean(q_T) | Std(q_T) |
|----------|--------------|-------------|-----------|----------|
| **Inventory** | 68.3545 (SE: 0.2735) | 8.6486 | 0.0090 (SE: 0.1638) | 5.1796 |
| **Symmetric** | 68.7571 (SE: 0.4203) | 13.2916 | 0.1490 (SE: 0.2735) | 8.6498 |

### Key Findings

- **Std(Profit) Reduction:** Inventory strategy has 0.6507x the variance of symmetric strategy
- **Std(q_T) Reduction:** Inventory strategy has 0.5988x the inventory variance of symmetric strategy
- **Mean Profit Difference:** -0.4026 (inventory - symmetric)

### Statistical Tests


#### Profit Variance Equality Tests

**F-test:**
- F-statistic: 2.3619
- p-value: 2.2204e-16
- Variance ratio (inv/sym): 0.4234

**Levene Test (Brown-Forsythe):**
- Statistic: 86.3512
- p-value: 3.8109e-20

**Bootstrap Confidence Intervals (95%, 10,000 resamples):**
- Mean difference (inv - sym): [-1.4061, 0.5909]
- Std ratio (inv / sym): [0.5947, 0.7120]
- Variance ratio (inv / sym): [0.3537, 0.5070]

#### Terminal Inventory (q_T) Variance Equality Tests

**F-test:**
- F-statistic: 2.7889
- p-value: 2.2204e-16
- Variance ratio (inv/sym): 0.3586

**Levene Test (Brown-Forsythe):**
- Statistic: 195.1484
- p-value: 2.2013e-42

**Bootstrap Confidence Intervals (95%, 10,000 resamples):**
- Mean difference (inv - sym): [-0.7650, 0.4930]
- Std ratio (inv / sym): [0.5620, 0.6371]
- Variance ratio (inv / sym): [0.3158, 0.4058]

---

## Experiment 3: High Risk Aversion (γ = 0.5) - Strong Variance Reduction

**Experiment ID:** exp_3  
**Risk Aversion (γ):** 0.5  
**Optimal Spread L(γ):** 1.1507  
**Number of Episodes:** 1000

### Summary Statistics

| Strategy | Mean(Profit) | Std(Profit) | Mean(q_T) | Std(q_T) |
|----------|--------------|-------------|-----------|----------|
| **Inventory** | 51.5335 (SE: 0.1499) | 4.7408 | -0.0090 (SE: 0.0630) | 1.9937 |
| **Symmetric** | 68.0654 (SE: 0.4293) | 13.5750 | 0.1750 (SE: 0.2892) | 9.1441 |

### Key Findings

- **Std(Profit) Reduction:** Inventory strategy has 0.3492x the variance of symmetric strategy
- **Std(q_T) Reduction:** Inventory strategy has 0.2180x the inventory variance of symmetric strategy
- **Mean Profit Difference:** -16.5319 (inventory - symmetric)

### Statistical Tests


#### Profit Variance Equality Tests

**F-test:**
- F-statistic: 8.1993
- p-value: 2.2204e-16
- Variance ratio (inv/sym): 0.1220

**Levene Test (Brown-Forsythe):**
- Statistic: 414.9977
- p-value: 5.6669e-84

**Bootstrap Confidence Intervals (95%, 10,000 resamples):**
- Mean difference (inv - sym): [-17.4373, -15.6391]
- Std ratio (inv / sym): [0.3219, 0.3796]
- Variance ratio (inv / sym): [0.1036, 0.1441]

#### Terminal Inventory (q_T) Variance Equality Tests

**F-test:**
- F-statistic: 21.0355
- p-value: 2.2204e-16
- Variance ratio (inv/sym): 0.0475

**Levene Test (Brown-Forsythe):**
- Statistic: 962.4889
- p-value: 7.9057e-173

**Bootstrap Confidence Intervals (95%, 10,000 resamples):**
- Mean difference (inv - sym): [-0.7670, 0.4060]
- Std ratio (inv / sym): [0.2047, 0.2324]
- Variance ratio (inv / sym): [0.0419, 0.0540]

---

