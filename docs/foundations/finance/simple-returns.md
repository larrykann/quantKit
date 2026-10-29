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
