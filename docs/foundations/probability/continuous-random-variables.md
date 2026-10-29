# Continuous Random Variables

Continuous random variables can take any value within a given range, with infinite possible values theoretically. Unlike discrete random variables, they cannot be counted conventionally and require integration for probability calculations.

## Key Characteristics

- Uncountable outcomes (infinite possible values in any interval)
- Can take any real value within their range
- No gaps between possible values
- Associated with a Probability Density Function (PDF)
- Both sample space $\Omega$ and event space $\mathcal{F}$ contain continuous elements
- The probability of any single exact value is zero: $P(X = x) = 0$

## Mathematical Properties

For a continuous random variable $X$ with PDF $f(x)$:

$$P(X = x) = 0 \text{ for any specific value } x$$

$$P(a \leq X \leq b) = \int_a^b f(x)dx$$

$$\int_{-\infty}^{\infty} f(x)dx = 1$$

Since the probability of any exact value is zero, we can only meaningfully calculate probabilities over intervals.

## Why P(X = x) = 0?

Consider a stock price that can be any value between $100 and $200. How many possible values exist in this range? Infinitely many. The probability of the stock being at exactly $150.00000000... (with infinite precision) is zero. However, the probability of being between $149.99 and $150.01 is non-zero.

## Examples

**Physical measurements:**

- Stock prices (can be any positive real number)
- Time intervals between events
- Temperature measurements
- Any measurement supporting floating-point arithmetic

**Financial examples:**

- Stock price: Could be $150.27, $150.271, $150.2718... (infinite precision theoretically)
- Return: Could be any value like 2.37%, -1.83%, 0.0001%
- Interest rate: 5.25%, 5.251%, 5.25123%...

## Financial Applications

**Price modeling:**

- Stock prices and asset values
- Foreign exchange rates
- Commodity prices
- Index levels

**Return analysis:**

- Daily, weekly, monthly returns
- Log returns for statistical modeling
- Portfolio returns

**Risk metrics:**

- Volatility measures
- Value at Risk (VaR)
- Expected shortfall

**Time-based variables:**

- Time until next market event
- Duration between trades
- Time to option expiration

## Probability Calculations

Unlike discrete variables where we can ask "What's the probability of getting exactly 3?", for continuous variables we must ask:

- "What's the probability of being between 3 and 4?"
- "What's the probability of being less than 5?"
- "What's the probability of exceeding 10?"

All require integration over an interval:

$$P(X > a) = \int_a^{\infty} f(x)dx$$

$$P(X < b) = \int_{-\infty}^b f(x)dx$$

## Contrast with Discrete Random Variables

| Aspect | Discrete | Continuous |
|--------|----------|------------|
| Possible values | Countable | Uncountable |
| Gaps | Yes | No |
| Single value probability | $P(X=x) > 0$ | $P(X=x) = 0$ |
| Probability function | PMF | PDF |
| Calculation method | Summation | Integration |
| Example | Dice roll {1,2,3,4,5,6} | Stock price (any value > 0) |

## Common Continuous Distributions in Finance

**Normal (Gaussian) distribution:**

- Models log returns
- Foundation of Black-Scholes
- Central to risk management

**Log-normal distribution:**

- Models stock prices
- Ensures prices stay positive
- Consistent with geometric Brownian motion

**Exponential distribution:**

- Time between events
- Waiting times in queuing models

**Student's t-distribution:**

- Heavy-tailed alternative to normal
- Better captures extreme market movements

## Computational Considerations

In practice, computer representations of continuous variables are discrete (finite precision). However, we treat them as continuous when:

- The number of possible values is very large
- The spacing between values is very small
- The underlying theoretical model is continuous

## References

Nag, A. (2024). *Stochastic Finance with Python* (Chapter 3). Apress.

Papoulis, A., & Pillai, S. U. (2002). *Probability, Random Variables and Stochastic Processes* (4th ed.). McGraw-Hill.
