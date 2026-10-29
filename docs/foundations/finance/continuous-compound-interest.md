# Continuous Compound Interest

**Continuous compound interest** represents the mathematical limit of discrete compound interest as the compounding frequency approaches infinity. Instead of discrete compounding periods, interest accrues continuously at every instant, resulting in the highest possible return for a given interest rate.

## Mathematical Formula

$$A = Pe^{rt}$$

where:

- $A$ = Final amount
- $P$ = Principal amount
- $e$ = Euler's number (≈ 2.71828)
- $r$ = Annual interest rate (as decimal)
- $t$ = Time in years

## Derivation from Discrete Compounding

Starting with discrete compound interest: $A = P\left(1 + \frac{r}{m}\right)^{mt}$

As $m \to \infty$:

$$\lim_{m \to \infty} P\left(1 + \frac{r}{m}\right)^{mt} = Pe^{rt}$$

This uses the fundamental limit: $\lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n = e$

## Key Characteristics

- Maximum possible return for given interest rate
- Exponential growth with natural logarithm base
- Theoretical upper bound for compound interest
- Extensively used in financial modeling and options pricing

## Financial Example

$10,000 invested at 5% annual rate for 4 years with continuous compounding:

$$A = 10,000 \times e^{0.05 \times 4} = 10,000 \times e^{0.2} = \$12,214.03$$

Compare to quarterly compounding: $10,000(1.0125)^{16} = \$12,202.55$

Continuous advantage: $11.48

## Modern Applications

- Options pricing models (Black-Scholes)
- Theoretical finance calculations
- Mathematical finance research
- Risk-free rate modeling in derivatives

## References

Fisher, I. (1930). *The Theory of Interest*. Macmillan.

Nag, A. (2024). *Stochastic Finance with Python* (Chapter 2). Apress.
