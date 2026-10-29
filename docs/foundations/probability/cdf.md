# Cumulative Distribution Function

The **Cumulative Distribution Function (CDF)** for a random variable $X$ is defined as:

$$F_X(x) = P(X \leq x)$$

representing the probability that $X$ will take a value less than or equal to $x$. It is crucial for describing the distribution of probabilities in statistics.

## Key Properties

- Always non-decreasing: if $x_1 < x_2$, then $F(x_1) \leq F(x_2)$
- Range is [0, 1]: $0 \leq F(x) \leq 1$ for all $x$
- Limits: $\lim_{x \to -\infty} F(x) = 0$ and $\lim_{x \to \infty} F(x) = 1$
- Right-continuous function
- Works for both discrete and continuous random variables

## Relationship to PDF and PMF

**For continuous random variables:**
The CDF is the integral of the PDF:

$$F(x) = \int_{-\infty}^x f(t)dt$$

Conversely, the PDF is the derivative of the CDF:

$$f(x) = \frac{dF(x)}{dx}$$

**For discrete random variables:**

The CDF is the sum of the PMF:

$$F(x) = \sum_{t \leq x} f(t)$$

## Computing Probabilities with CDF

The CDF allows easy computation of interval probabilities:

$$P(a < X \leq b) = F(b) - F(a)$$

This is particularly useful because many statistical tables and software functions provide CDFs rather than PDFs or PMFs.

## Standard Normal CDF

For the standard normal distribution, the CDF is often denoted as $\Phi(z)$:

$$\Phi(z) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^z e^{-\frac{t^2}{2}}dt$$

This integral has no closed-form solution, so it must be computed using:

- Numerical integration
- Polynomial approximations
- Statistical tables
- Software libraries

### Polynomial Approximation

A common approach uses a polynomial approximation for computational efficiency. The process involves:

1. Computing a scaling factor based on the z-score:

$$t = \frac{1}{1 + 0.2316419 \times |z|}$$

2. Evaluating a polynomial that approximates the integral:

$$\text{poly} = (1.330274429 \times t - 1.821255978) \times t + 1.781477937 \times t - 0.356563782 \times t + 0.319381530 \times t$$

3. Combining with the PDF for the final CDF value:

$$F(z) = 
\begin{cases} 
1.0 - \text{pdf}(z) \times \text{poly} & \text{if } z > 0 \\
\text{pdf}(z) \times \text{poly} & \text{if } z \leq 0
\end{cases}$$

The polynomial coefficients are chosen to maximize precision, achieving accuracy up to $7.5 \times 10^{-8}$, suitable for most statistical applications.

## Financial Applications

**Risk Management:**

- Value at Risk (VaR): $P(\text{Loss} \leq \text{VaR}_\alpha) = \alpha$
- Probability of portfolio loss exceeding threshold
- Credit default probabilities

**Option Pricing:**

- Black-Scholes formula uses normal CDF ($\Phi$) extensively
- Computing probabilities of finishing in-the-money

**Statistical Testing:**

- Hypothesis tests use CDF to compute p-values
- Confidence interval construction
- Significance testing for trading strategies

## Advantages of Using CDF

1. **Unified framework**: Works for both discrete and continuous variables
2. **Probability calculations**: Easy to compute $P(X \leq x)$ directly
3. **Quantile functions**: Inverse CDF gives quantiles and percentiles
4. **Comparison**: Can compare distributions visually using CDF plots
5. **Numerical stability**: Often more stable computationally than PDF

## References

Nag, A. (2024). *Stochastic Finance with Python* (Chapter 3). Apress.

Papoulis, A., & Pillai, S. U. (2002). *Probability, Random Variables and Stochastic Processes* (4th ed.). McGraw-Hill.

Masters, T. (2020). *Statistically Sound Indicators for Financial Market Prediction*. CreateSpace Independent Publishing Platform.
