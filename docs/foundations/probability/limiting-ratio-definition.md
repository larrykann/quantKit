# Probability - Limiting Ratio Definition

When we can't determine the exact size of the sample space or when outcomes are too complex to count directly, we must estimate probability through repeated experimentation. For this, we can use the limiting ratio definition of probability:

$$P(E) = \lim_{N \to \infty} \frac{c}{N}$$

where $c$ is the number of outcomes supporting event $E$ and $N$ is the number of trials conducted.

## Theoretical Significance

This expression has experimental rather than mathematical significance, as the true probability would theoretically be found only by repeating the trials of a random experiment an infinite number of times.

In practice, the probability value $P(E)$ lies between 0 and 1:

$$0 \leq P(E) \leq 1$$

## When This Definition is Needed

- The sample space is infinite or uncountable
- Analytic methods for exploring the complete sample space are impossible
- Working with continuous data where classical counting doesn't apply
- Complex experiments where complete enumeration is impractical

## Financial Context

Financial variables present unique challenges:

**Uncountable sample spaces:**

- How many possible values can a stock index take? Theoretically infinite
- Stock prices are continuous and can take any positive real value
- Returns can range across continuous intervals

**Practical necessity:**

Since we cannot enumerate all possible stock prices or write down the complete sample space, we must:

1. Repeat observations as trials (collect historical data)
2. Record the supporting events
3. Use frequency of occurrence to estimate probabilities

## Relationship to Empirical Probability

The limiting ratio definition provides the theoretical foundation for empirical probability estimation:

$$P(E) \approx \frac{\text{observed occurrences of } E}{\text{total observations}}$$

As the number of observations increases, this ratio converges to the true probability (by the Law of Large Numbers).

## Example: Estimating Probability of Positive Returns

To estimate the probability that a stock has a positive daily return:

1. Collect N days of historical data
2. Count c days with positive returns
3. Estimate: $P(\text{positive return}) \approx \frac{c}{N}$
4. As N increases, the estimate improves

## Advantages Over Classical Definition

- Applicable to continuous and infinite sample spaces
- Works when outcomes are not equally likely
- Provides foundation for statistical inference
- Directly connects to real-world data collection

## References

Nag, A. (2024). *Stochastic Finance with Python* (Chapter 3). Apress.

von Mises, R. (1957). *Probability, Statistics and Truth*. Dover Publications.
