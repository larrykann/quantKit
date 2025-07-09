# Multiperiod Simple Returns

**Multiperiod simple returns** extend the concept of single-period returns to measure performance over multiple time intervals. These returns can be decomposed into products of individual period returns, providing insight into how investment gains compound over extended holding periods.

## Mathematical Formula

For a τ-period return:
$$R_t(\tau) = \frac{S_t}{S_{t-\tau}} - 1$$

where:
- $R_t(\tau)$ = τ-period return ending at time $t$
- $S_t$ = Asset price at time $t$
- $S_{t-\tau}$ = Asset price τ periods earlier
- $\tau$ = Number of periods (can be days, weeks, months, etc.)

## Decomposition into Single-Period Returns

A key property of multiperiod simple returns is their multiplicative decomposition:

$$1 + R_t(\tau) = \frac{S_t}{S_{t-\tau}} = \frac{S_t}{S_{t-1}} \times \frac{S_{t-1}}{S_{t-2}} \times \cdots \times \frac{S_{t-\tau+1}}{S_{t-\tau}}$$

$$= (1 + R_t)(1 + R_{t-1}) \cdots (1 + R_{t-\tau+1})$$

Therefore:
$$R_t(\tau) = \prod_{i=0}^{\tau-1}(1 + R_{t-i}) - 1$$

## Key Characteristics

- **Multiplicative structure** - multiperiod returns are products, not sums, of single-period returns
- **Path dependent** - the sequence of individual returns matters for compounding
- **Non-linear aggregation** - cannot simply add single-period returns to get multiperiod returns
- **Compounding effect** - gains and losses compound over time

## Financial Examples

**Example 1: Weekly return from daily returns**
- Monday: 2% gain → $R_1 = 0.02$
- Tuesday: 1% loss → $R_2 = -0.01$  
- Wednesday: 3% gain → $R_3 = 0.03$

Weekly return:
$$R_{week} = (1.02)(0.99)(1.03) - 1 = 1.0405 - 1 = 0.0405 = 4.05\%$$

**Example 2: Quarterly return**
Monthly returns: 5%, -2%, 4%
$$R_{quarter} = (1.05)(0.98)(1.04) - 1 = 1.0703 - 1 = 7.03\%$$

Note: Simple addition would incorrectly give 7%.

## Implementation Examples

### Python

```python
def calculate_multiperiod_return(single_period_returns):
    """Calculate multiperiod return from sequence of single-period returns"""
    if not single_period_returns:
        return 0.0
    
    cumulative_factor = 1.0
    for return_rate in single_period_returns:
        cumulative_factor *= (1 + return_rate)
    
    return cumulative_factor - 1.0

def decompose_multiperiod_return(prices, start_index, end_index):
    """Decompose multiperiod return into individual period components"""
    if start_index >= end_index or end_index >= len(prices):
        return None, None
    
    # Calculate individual period returns
    period_returns = []
    for i in range(start_index + 1, end_index + 1):
        period_return = (prices[i] - prices[i-1]) / prices[i-1]
        period_returns.append(period_return)
    
    # Calculate multiperiod return two ways for verification
    direct_multiperiod = (prices[end_index] - prices[start_index]) / prices[start_index]
    composed_multiperiod = calculate_multiperiod_return(period_returns)
    
    return period_returns, (direct_multiperiod, composed_multiperiod)

def analyze_multiperiod_returns(prices, periods_list=[5, 10, 22]):
    """Analyze returns over different time horizons"""
    print("Multiperiod Return Analysis")
    print("=" * 50)
    
    for period_length in periods_list:
        if len(prices) > period_length:
            multiperiod_returns = []
            
            for i in range(period_length, len(prices)):
                period_return = (prices[i] - prices[i - period_length]) / prices[i - period_length]
                multiperiod_returns.append(period_return)
            
            if multiperiod_returns:
                avg_return = sum(multiperiod_returns) / len(multiperiod_returns)
                volatility = (sum([(r - avg_return)**2 for r in multiperiod_returns]) / len(multiperiod_returns))**0.5
                
                print(f"{period_length:2d}-period returns:")
                print(f"  Average: {avg_return:8.4f} ({avg_return*100:6.2f}%)")
                print(f"  Volatility: {volatility:6.4f} ({volatility*100:4.2f}%)")
                print(f"  Observations: {len(multiperiod_returns)}")
                print()

# Example usage
stock_prices = [100, 102, 99, 105, 103, 108, 110, 107, 112, 115, 118]
daily_returns = [0.02, -0.0294, 0.0606, -0.0190, 0.0485, 0.0185, -0.0273, 0.0467, 0.0268, 0.0261]

print("Stock Prices:", stock_prices[:6])
print("Daily Returns:", [f"{r:.4f}" for r in daily_returns[:5]])

# Calculate 5-day return
five_day_return = calculate_multiperiod_return(daily_returns[:5])
print(f"\n5-day multiperiod return: {five_day_return:.4f} ({five_day_return*100:.2f}%)")

# Verify with direct calculation
direct_calc = (stock_prices[5] - stock_prices[0]) / stock_prices[0]
print(f"Direct calculation verify: {direct_calc:.4f} ({direct_calc*100:.2f}%)")

# Analyze different horizons
analyze_multiperiod_returns(stock_prices, [2, 3, 5])
```

## Practical Applications

- **Performance attribution** - understanding contribution of different periods
- **Risk measurement** - longer horizons often show different risk profiles
- **Investment horizon analysis** - comparing short vs. long-term performance
- **Backtesting strategies** - evaluating performance over various time frames
- **Benchmark comparison** - standardizing returns across different time periods

## Relationship to Compounding

Multiperiod returns demonstrate the power of compounding:
- **Positive compounding** - gains build upon previous gains
- **Negative compounding** - losses compound and amplify downside risk
- **Sequence risk** - the order of returns affects final outcomes

**Example of sequence risk:**
- Scenario A: +20%, -10% → Final: (1.20)(0.90) - 1 = 8%
- Scenario B: -10%, +20% → Final: (0.90)(1.20) - 1 = 8%

Both sequences yield the same result, but interim volatility differs.

## Advantages and Limitations

**Advantages:**
- Intuitive interpretation of cumulative performance
- Direct relationship to wealth creation
- Natural compounding structure

**Limitations:**
- **Non-additivity** - cannot sum across securities or time periods
- **Mathematical complexity** - products are harder to work with than sums
- **Distribution properties** - more complex statistical behavior than single-period returns

These limitations motivate the use of log returns in many financial models, as they convert multiplicative relationships into additive ones.

## References

Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *Journal of Finance*, 25(2), 383-417.

Nag, A. (2024). *Stochastic Finance with Python* (Chapter 2). Apress.
