# Discrete Random Variables

Discrete random variables take countable, distinct values such as integers or categories. The key characteristic is that you can list out all possible values the variable can take, even if that list is very long.

## Key Properties

- Values can be listed out completely, even if the list is very long
- There are gaps between possible values (like 1, 2, 3 - nothing between them)
- Often used for counting things or categorizing outcomes
- Work with Probability Mass Function (PMF) for calculations
- Each specific value has a non-zero probability of occurring
- All probabilities sum to exactly 1

## Mathematical Characteristics

For a discrete random variable $X$ taking values $x_1, x_2, x_3, \ldots$:

$$P(X = x_i) > 0 \text{ for each possible value } x_i$$

$$\sum_{\text{all } x_i} P(X = x_i) = 1$$

The probability distribution is fully specified by listing each possible value and its associated probability.

## Examples

**Simple discrete variables:**

- Number of heads in coin tosses: {0, 1, 2, 3, ...}
- Dice roll outcome: {1, 2, 3, 4, 5, 6}
- Binary events: {0, 1} for {Failure, Success}

**Poker example:**

Number of pocket pairs dealt in a Texas Hold'em session: {0, 1, 2, 3, ...}

- If playing 100 hands, possible values are 0 through 100
- Each value has a specific probability based on binomial distribution
- Can calculate exact probability of getting exactly 5 pocket pairs

## Financial Applications

**Counting applications:**

- Number of significant price jumps in a stock within a timeframe
- Count of trading days with volume above a threshold
- Number of dividends paid per year
- Number of defaults in a bond portfolio

**Categorical applications:**

- Credit rating categories: {AAA, AA, A, BBB, BB, B, CCC, CC, C, D}
- Trade direction: {Buy, Sell, Hold}
- Market regime: {Bull, Bear, Sideways}
- Option exercise decision: {Exercise, Hold, Expire}

## Contrast with Continuous Random Variables

**Discrete:**

- Countable outcomes
- Gaps between values
- $P(X = x)$ can be non-zero
- Uses PMF

**Continuous:**

- Uncountable outcomes
- No gaps (any value in range possible)
- $P(X = x) = 0$ for any specific value
- Uses PDF

## Common Discrete Distributions in Finance

**Binomial distribution:**

- Number of successful outcomes in fixed trials
- Example: Number of profitable trades out of N trades

**Poisson distribution:**

- Number of events in fixed time interval
- Example: Number of large price jumps per month

**Bernoulli distribution:**

- Single binary outcome
- Example: Will stock price increase today? {Yes, No}

## References

Nag, A. (2024). *Stochastic Finance with Python* (Chapter 3). Apress.

Papoulis, A., & Pillai, S. U. (2002). *Probability, Random Variables and Stochastic Processes* (4th ed.). McGraw-Hill.
