# Simple Return

**Simple return** measures the percentage change in an asset's value over a specific time period, representing the fundamental metric for evaluating investment performance. It calculates the ratio of profit or loss relative to the initial investment amount.

## Mathematical Formula

$$R_t = \frac{S_t - S_{t-1}}{S_{t-1}} = \frac{S_t}{S_{t-1}} - 1$$

where:
- $R_t$ = Simple return at time $t$
- $S_t$ = Asset price at time $t$
- $S_{t-1}$ = Asset price at time $t-1$

## Alternative Formulations

**Forward-looking return:**
$$R_t = \frac{S_{t+1} - S_t}{S_t}$$

**Periodic return with time interval $\tau$:**
$$R_t(\tau) = \frac{S_t}{S_{t-\tau}} - 1$$

## Key Characteristics

- Expressed as decimal or percentage
- Bounded below by -1 (complete loss)
- Unbounded above (theoretically infinite gains)
- Additive across different assets in portfolio context
- Intuitive interpretation as profit/loss percentage

## Financial Example

Consider Apple stock trading at $150 on Monday and $159 on Friday:

$$R_{Friday} = \frac{159 - 150}{150} = \frac{9}{150} = 0.06 = 6\%$$

This represents a 6% gain over the holding period.

**Quarterly return example:**
- Q1 ending price: $200
- Q4 ending price: $185
- Quarterly return: $\frac{185 - 200}{200} = -0.075 = -7.5\%$

## Implementation Examples

### Python

```python
def calculate_simple_return(current_price, previous_price):
    """Calculate simple return between two price points"""
    return (current_price - previous_price) / previous_price

def calculate_periodic_returns(prices, period=1):
    """Calculate returns for a series of prices with specified period"""
    returns = []
    
    for i in range(period, len(prices)):
        current_price = prices[i]
        previous_price = prices[i - period]
        return_value = calculate_simple_return(current_price, previous_price)
        returns.append(return_value)
    
    return returns

def analyze_returns(prices, labels=None):
    """Analyze returns for different time periods"""
    periods = {'Daily': 1, 'Weekly': 5, 'Monthly': 22}
    
    if labels is None:
        labels = [f"Day_{i}" for i in range(len(prices))]
    
    print("Return Analysis:")
    print("-" * 40)
    
    for period_name, period_length in periods.items():
        if len(prices) > period_length:
            returns = calculate_periodic_returns(prices, period_length)
            avg_return = sum(returns) / len(returns) if returns else 0
            print(f"{period_name:8} Average Return: {avg_return:8.4f} ({avg_return*100:6.2f}%)")

# Example usage
stock_prices = [100.0, 102.5, 98.0, 105.0, 103.2, 107.8, 110.5]
daily_returns = calculate_periodic_returns(stock_prices, 1)

print("Stock Prices:", stock_prices)
print("Daily Returns:", [f"{r:.4f}" for r in daily_returns])
print()

analyze_returns(stock_prices)

# Compare with simple interest return
principal = 1000.0
interest_rate = 0.05
time_period = 1.0
simple_interest_return = interest_rate * time_period

print(f"\nSimple Interest Return (5% for 1 year): {simple_interest_return:.4f}")
print(f"Equivalent to: {simple_interest_return*100:.2f}%")
```

## Return vs. Interest Rate Comparison

For deterministic assets (bonds, savings accounts):
$$R_t^{Simple\ Interest} = rt$$

where $r$ is the annual interest rate and $t$ is the time period.

**Example:** A 3% annual savings account yields exactly 3% simple return over one year, regardless of market conditions.

**Contrast with stocks:** Returns are uncertain and can be positive or negative, making them suitable for stochastic modeling.

## Applications in Finance

- **Performance measurement** for individual securities
- **Benchmark comparison** against market indices
- **Portfolio allocation** decisions based on historical returns
- **Risk assessment** through return volatility analysis
- **Investment strategy** evaluation over different time horizons

## Limitations

- **Lower bound constraint** at -100% creates mathematical complications
- **Not time-additive** - returns across periods don't sum directly
- **Asymmetric treatment** of gains vs. losses
- **Scale dependency** - difficult to aggregate across different asset classes

These limitations lead many financial models to prefer log returns for mathematical convenience and superior statistical properties.

## References

Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *Journal of Finance*, 25(2), 383-417.

Nag, A. (2024). *Stochastic Finance with Python* (Chapter 2). Apress.
