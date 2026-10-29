# Probability Density Function

The **Probability Density Function (PDF)** describes the density of probabilities across values that a continuous random variable can take. Unlike a Probability Mass Function, which gives exact probabilities for discrete values, the PDF represents probability density rather than probability itself.

## Key Properties

- Applies only to continuous random variables
- PDF values can exceed 1 (represents density, not probability)
- The probability of any exact value P(X = x) = 0
- Total area under the PDF curve equals 1
- Probabilities are calculated over intervals using integration

## Mathematical Definition

For a continuous random variable X with PDF f(x):

$$f(x) \geq 0 \text{ and } \int_{-\infty}^{\infty} f(x)dx = 1$$

## Calculating Probabilities

To find the probability that X falls between values a and b:

$$P(a \leq X \leq b) = \int_a^b f(x)dx$$

## Financial Example

Consider a stock price that can take any real value between $100-$200. The PDF might show high density around $150 (where the stock commonly trades) and lower density at the extremes. Even though some PDF values might exceed 1.0, the total area under the curve equals 1.

## Important Examples

### Gaussian (Normal) Distribution

The **Gaussian** or **Normal distribution** is the most important continuous distribution in finance. For a random variable with mean μ and variance σ²:

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

For the **standard normal** distribution (μ = 0, σ = 1):

$$f(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}$$

This function represents the bell curve, where the highest point is at the mean, and the probability density drops off as you move away from the mean.

**Why it matters in finance:**

- Log returns are often assumed to be normally distributed
- Foundation of Black-Scholes option pricing model
- Central Limit Theorem ensures many aggregated quantities approximate normal distributions
- Used extensively in risk management (VaR calculations)
- Underpins many statistical tests and confidence intervals

### Other Common Distributions

- **Exponential**: Models time between events (e.g., time between trades)
- **Uniform**: Equal probability density across a range
- **Log-normal**: Models stock prices (if log returns are normal)
- **Student's t**: Heavy-tailed alternative to normal for modeling extreme returns

## References

Nag, A. (2024). *Stochastic Finance with Python* (Chapter 3). Apress.

Hogg, R. V., & Craig, A. T. (1995). *Introduction to Mathematical Statistics* (5th ed., Chapter 3). Prentice Hall.

Masters, T. (2020). *Statistically Sound Indicators for Financial Market Prediction*. CreateSpace Independent Publishing Platform.
