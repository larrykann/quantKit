# Probability -- Classic Definition

The classical definition of probability states that if c is the number of outcomes supporting an event E and N is the total number of possible outcomes, then:

$$P(E) = \frac{c}{N}$$

More formally, using set-theoretic notation:

$$P(E) = \frac{n(E)}{n(\Omega)}$$

where n(E) is the number of elements in event E and n(Ω) is the number of elements in the sample space.

## Key Requirements

- All outcomes in the sample space are equally likely
- The sample space is finite and well-defined
- Events are clearly defined subsets of the sample space

## Texas Hold'em Example

In Texas Hold'em, what's the probability of being dealt pocket aces (two aces)?

- Event E = {getting two aces}, so n(E) = 6 possible combinations
- Sample space Ω = {all possible two-card hands}, so n(Ω) = 1,326 combinations
- Therefore: P(E) = 6/1,326 ≈ 0.0045 or about 0.45%

## Limitations

The classical definition only applies when:

1. The sample space is finite
2. All outcomes are equally likely
3. The complete sample space can be enumerated

For continuous variables, infinite sample spaces, or unequal probabilities, other definitions are needed.

## Financial Applications

**Discrete scenarios with known, equal probabilities:**

- Lottery or raffle outcomes
- Simple game theory payoffs
- Discrete event simulations with uniform probabilities

**Limited applicability in real finance:**
Most financial applications involve continuous variables (prices, returns) or unequal probabilities (market movements), requiring the limiting ratio definition or empirical estimation instead.

## Historical Context

This definition, often called the "classical" or "Laplacian" approach, provides the foundation for probability theory when dealing with finite, equally likely outcomes. It works well for games of chance but is insufficient for most real-world financial modeling.

## References

Nag, A. (2024). *Stochastic Finance with Python* (Chapter 3). Apress.

Kolmogorov, A. N. (1950). *Foundations of the Theory of Probability*. Chelsea Publishing.
