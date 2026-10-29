# Probability Mass Function

The **Probability Mass Function (PMF)** is a function that gives the probability that a discrete random variable takes on a specific value. Unlike continuous variables, discrete variables have countable outcomes where we can assign exact probabilities to each possible value.

## Mathematical Definition

For a discrete random variable X, the PMF is denoted as f(x) where:

$$0 \leq f(x) \leq 1 \text{ and } \sum_{x} f(x) = 1$$

## Key Properties

- Only applies to discrete random variables
- Assigns exact probabilities to specific values
- All probabilities must sum to 1 across all possible outcomes
- Each individual probability is between 0 and 1
- Creates a bar chart appearance when plotted

## Texas Hold'em Example

Consider the random variable X = "number of pocket pairs dealt to you in exactly 50 hands." Since the probability of being dealt any pocket pair on a single hand is 1/17 ≈ 0.0588, this follows a binomial distribution.

The PMF would assign probabilities like:

- f(0) = 0.0302 (3.02% chance of zero pocket pairs in 50 hands)
- f(1) = 0.0915 (9.15% chance of exactly one pocket pair)
- f(2) = 0.1373 (13.73% chance of exactly two pocket pairs)
- f(3) = 0.1360 (13.60% chance of exactly three pocket pairs)
- f(4) = 0.1003 (10.03% chance of exactly four pocket pairs)
- And so on for 5, 6, 7... up to 50 pocket pairs

Notice that while we show only the first few values, all probabilities from f(0) to f(50) would sum to exactly 1.00, satisfying the PMF requirement.

## Contrast with PDF

Unlike a Probability Density Function, which deals with continuous variables and probability densities, the PMF gives exact probabilities for discrete outcomes. You can't have "1.5 pocket pairs" - the values are countable and distinct.

## Financial Applications

- Number of dividend payments per year: {0, 1, 2, 3, 4}
- Credit rating categories: {AAA, AA, A, BBB, BB, B, CCC}
- Number of trading days with volume above threshold
- Count of significant price jumps in a given period
- Number of defaults in a bond portfolio
- Daily direction changes (up/down/unchanged)

## References

Nag, A. (2024). *Stochastic Finance with Python* (Chapter 3). Apress.

Papoulis, A., & Pillai, S. U. (2002). *Probability, Random Variables and Stochastic Processes* (4th ed., Chapter 4). McGraw-Hill.
