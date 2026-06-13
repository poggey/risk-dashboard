# Risk Analytics Dashboard

Interactive Streamlit dashboard for portfolio risk and performance analysis.

## Status
In development.

## Scenarios Tab (Stress Test)

Simulates how your portfolio would behave during a market crisis by combining **factor shocks** with **drawdown anatomy**.

### Core Concept

Instead of arbitrarily saying "what if AAPL drops 40%", this approach:
1. Shocks underlying **market factors** (not individual assets)
2. Propagates those shocks to each asset based on its **factor sensitivities**
3. Simulates the crisis **over time** with realistic phases (crash → bottom → recovery)

### Inputs

#### Factor Shocks (3 sliders)

| Factor | What it represents | How it affects assets |
|--------|-------------------|----------------------|
| Market Shock | Broad equity decline (%) | Multiplied by each asset's **beta** |
| Interest Rate Shock | Rate increase (basis points) | Higher volatility assets are more rate-sensitive |
| Volatility Spike | VIX level during crisis | Higher VIX = additional negative impact |

#### Crisis Shape (3 options)

| Shape | Crash | Bottom | Recovery | Real-world example |
|-------|-------|--------|----------|-------------------|
| V-Shape | 4 weeks | 0 weeks | 12 weeks | COVID March 2020 |
| U-Shape | 8 weeks | 16 weeks | 24 weeks | 2008 Financial Crisis |
| L-Shape | 8 weeks | 52 weeks | 0 weeks | Japan 1990s |

### Calculations

#### 1. Calculate each asset's market beta
```
beta = covariance(asset_returns, benchmark_returns) / variance(benchmark_returns)
```
Beta measures how much the asset moves relative to the market. Beta > 1 = more volatile than market.

#### 2. Estimate rate sensitivity
```
rate_sensitivity = -volatility × 0.5
```
Higher volatility assets are assumed to be more negatively affected by rate shocks (simplified proxy).

#### 3. Calculate total shock per asset
```
asset_shock = (beta × market_shock) + (rate_sensitivity × rate_shock/10000) + (vix_impact)
```
Where `vix_impact = -0.002 × (vix_level - 20)`

#### 4. Calculate portfolio shock
```
portfolio_shock = Σ (asset_shock × asset_weight)
```
Weighted sum of individual asset shocks.

### Timeline Chart Construction

The chart shows your portfolio's journey through a simulated crisis:

```
Portfolio Value
     │
 1.0 ┤━━━━━━━━━━━━┓                                    ╭━━━━━
     │   Actual    ┃                                  ╱
     │  (3 months) ┃ Crash                          ╱
     │             ┃  phase                       ╱  Recovery
     │             ┃    ↘                       ╱    phase
 0.7 ┤              ┗━━━━━━━━┳━━━━━━━━━━━━━━━━━╱
     │                       ┃   Bottom phase
     │                       ┃   (oscillates)
     ├─────────────┼─────────┼─────────────────┼──────────→ Time
           Pre     │  Crash  │     Bottom      │  Recovery
```

#### Step-by-step:

**1. Pre-shock (blue line)**
- Takes the last 63 trading days (~3 months) of your actual portfolio returns
- Calculates cumulative returns: `(1 + daily_return).cumprod()`
- Normalises to start at 1.0

**2. Crash phase (red shaded region)**
- Duration: depends on shape (V=4wks, U=8wks, L=8wks)
- Values: `np.linspace(1.0, 1.0 + portfolio_shock/100, crash_days)`
- This creates a straight line from current value down to the bottom
- Example: -30% shock → line from 1.0 to 0.7

**3. Bottom phase (orange shaded region)**
- Duration: depends on shape (V=0wks, U=16wks, L=52wks)
- Values: oscillates around the bottom with random noise
```python
noise = np.random.normal(0, 0.005, bottom_days)
bottom_values = bottom_base + np.cumsum(noise)
```
- This simulates the choppy, uncertain consolidation period

**4. Recovery phase (green shaded region)**
- Duration: depends on shape (V=12wks, U=24wks, L=0wks)
- Values: `np.linspace(recovery_start, 1.0, recovery_days)`
- Straight line from bottom back to original value (full recovery)

**5. Connecting to actual data**
- The scenario is scaled to connect smoothly with the pre-shock period:
```python
scenario_scaled = scenario_values * pre_shock_cumulative.iloc[-1]
```
- Dates continue from where actual data ends

### Output

- **Table**: Each asset's beta, estimated shock, and contribution to portfolio loss
- **Key stats**: Max drawdown, time to bottom, time at bottom, recovery time, total time underwater
- **Timeline chart**: Visual showing actual pre-shock data → simulated crash → bottom → recovery with colour-coded phases
- **Interpretation**: Summary of severity and what to expect

### Limitations

- Rate sensitivity is a simplified proxy (uses volatility, not actual duration/rate exposure)
- Recovery assumes linear path back to original value (reality is messier)
- Bottom phase uses random noise (seeded for reproducibility)
- Does not model correlation changes during crisis (correlations typically spike toward 1)

